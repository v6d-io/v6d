#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os
from typing import Any

from airflow.configuration import conf
from airflow.models.xcom import BaseXCom
from airflow.utils.session import provide_session
import pendulum
from sqlalchemy.orm import Session, reconstructor

import vineyard

logger = logging.getLogger('vineyard')


def _resolve_vineyard_xcom_options():
    options = {}
    if conf.has_option('vineyard', 'persist'):
        options['persist'] = conf.getboolean('vineyard', 'persist')
    else:
        options['persist'] = False
    if conf.has_option('vineyard', 'ipc_socket'):
        options['ipc_socket'] = conf.get('vineyard', 'ipc_socket')
    else:
        if 'VINEYARD_IPC_SOCKET' in os.environ:
            options['ipc_socket'] = os.environ['VINEYARD_IPC_SOCKET']
        else:
            raise RuntimeError("Failed to find vineyard IPC socket configuration, " +
                               "please configure it using the environment variable " +
                               "$VINEYARD_IPC_SOCKET, or via airfow's vineyard.ipc_socket configuration.")
    return options


class VineyardXCom(BaseXCom):
    """
    Custom Backend Serving to use Vineyard.

    Setup your airflow environment by specifying the following
    the environment varable:

        export AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom
    """

    __options = None

    @classmethod
    def options(cls):
        if cls.__options is None:
            cls.__options = _resolve_vineyard_xcom_options()
        return cls.__options

    @reconstructor
    def init_on_load(self):
        """
        Called by the ORM after the instance has been loaded from the DB or otherwise reconstituted
        i.e automatically deserialize Xcom value when loading from DB.
        """
        self.value = super(VineyardXCom, self).init_on_load()

    @classmethod
    @provide_session
    def set(cls, key, value, execution_date, task_id, dag_id, session=None):
        """
        Store an XCom value.
        :return: None
        """
        session.expunge_all()

        value = VineyardXCom.serialize_value(value)

        # remove any duplicate XComs
        query = session.query(cls).filter(cls.key == key, cls.execution_date == execution_date, cls.task_id == task_id,
                                          cls.dag_id == dag_id)
        targets = []
        for result in query.with_entities(VineyardXCom.value):
            targets.append(vineyard.ObjectID(BaseXCom.deserialize_value(result)))
        if targets:
            logger.info("Drop duplicates from vineyard: %s", targets)
            try:
                client = vineyard.connect(cls.options()['ipc_socket'])
                client.delete(targets)
            except Exception as e:
                logger.error('Failed to drop duplicates from vineyard: %s', e)

        # step 2: remove from the underlying xcom db
        query.delete()
        session.commit()

        # insert new XCom
        session.add(VineyardXCom(key=key, value=value, execution_date=execution_date, task_id=task_id, dag_id=dag_id))
        session.commit()

    @classmethod
    @provide_session
    def delete(cls, xcoms, session=None):
        """Delete Xcom"""
        if isinstance(xcoms, VineyardXCom):
            xcoms = [xcoms]
        targets = []
        for xcom in xcoms:
            if not isinstance(xcom, VineyardXCom):
                raise TypeError(f'Expected XCom; received {xcom.__class__.__name__}')
            if xcom.value:
                targets.append(vineyard.ObjectID(BaseXCom.deserialize_value(xcom)))
            session.delete(xcom)
        logger.info("Drop from vineyard: %s", targets)
        try:
            client = vineyard.connect(cls.options()['ipc_socket'])
            client.delete(targets)
        except Exception as e:
            logger.error('Failed to drop from vineyard: %s', e)
        session.commit()

    @classmethod
    @provide_session
    def clear(
        cls,
        execution_date: pendulum.DateTime,
        dag_id: str,
        task_id: str,
        session: Session = None,
    ) -> None:
        query = session.query(cls).filter(
            cls.dag_id == dag_id,
            cls.task_id == task_id,
            cls.execution_date == execution_date,
        )
        targets = []
        for result in query.with_entities(VineyardXCom.value):
            targets.append(vineyard.ObjectID(BaseXCom.deserialize_value(result)))
        if targets:
            logger.info("Drop from vineyard: %s", targets)
            try:
                client = vineyard.connect(cls.options()['ipc_socket'])
                client.delete(targets)
            except Exception as e:
                logger.error('Failed to drop from vineyard: %s', e)
        query.delete()

    @staticmethod
    def serialize_value(value: Any):
        client = vineyard.connect(VineyardXCom.options()['ipc_socket'])
        value_id = client.put(value)
        if VineyardXCom.options()['persist']:
            client.persist(value_id)
        logger.debug("serialize_value: %s -> %r", value, value_id)
        return BaseXCom.serialize_value(repr(value_id))

    @staticmethod
    def deserialize_value(result: "XCom") -> Any:
        value = BaseXCom.deserialize_value(result)
        vineyard_value = VineyardXCom.post_resolve_value(result, value)
        logger.debug("deserialize_value: %s ->  %s -> %s", result, value, vineyard_value)
        return vineyard_value

    @staticmethod
    @provide_session
    def post_resolve_value(result: "XCom", value: Any, session: Session = None) -> Any:
        ''' The :code:`post_resolve_value` runs before the return the value to the
            operators to prepare necessary input data for the task.

            The post resolution will fill-up the occurrence if remote objects by
            of :code:`VineyardObjectRef` with the actual (remote) value by triggering
            a migration.

            It will also record the migrated xcom value into the db as well to make
            sure it can be dropped properly.
        '''
        client = vineyard.connect(VineyardXCom.options()['ipc_socket'])
        object_id = vineyard.ObjectID(value)

        meta = client.get_meta(object_id)
        if meta.islocal:
            return client.get(object_id)

        # migration
        logger.debug('start migration: %r')
        target_id = client.migrate(object_id)
        logger.debug('finish migration: %r -> %r', object_id, target_id)

        # TODO: should we record the replicated XCom into the db ?
        # session.add(VineyardXCom(...))
        # session.commit()

        return client.get(target_id)


__all__ = [
    'VineyardXCom',
]
