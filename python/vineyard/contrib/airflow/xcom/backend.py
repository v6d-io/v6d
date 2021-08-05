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
from typing import Any

from airflow.models.xcom import BaseXCom
from airflow.utils.session import provide_session
import pendulum
from sqlalchemy.orm import Session, reconstructor

import vineyard

logger = logging.getLogger('vineyard')


class VineyardXCom(BaseXCom):
    """
    Custom Backend Serving to use Vineyard.

    Setup in your airflow Dockerfile with the following lines: ::

        FROM quay.io/astronomer/ap-airflow:2.0.2-1-buster-onbuild
        USER root
        RUN pip uninstall astronomer-airflow-version-check -y
        USER astro
        ENV AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom
    """
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
                client = vineyard.connect()
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
            client = vineyard.connect()
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
                client = vineyard.connect()
                client.delete(targets)
            except Exception as e:
                logger.error('Failed to drop from vineyard: %s', e)
        query.delete()

    @staticmethod
    def serialize_value(value: Any):
        client = vineyard.connect()
        value_id = repr(client.put(value))
        logger.debug("serialize_value: %s -> %s", value, value_id)
        return BaseXCom.serialize_value(value_id)

    @staticmethod
    def deserialize_value(result: "XCom") -> Any:
        value = BaseXCom.deserialize_value(result)
        client = vineyard.connect()
        vineyard_value = client.get(vineyard.ObjectID(value))
        logger.debug("deserialize_value: %s ->  %s -> %s", result, value, vineyard_value)
        return vineyard_value
