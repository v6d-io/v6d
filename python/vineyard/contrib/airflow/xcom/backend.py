#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

import datetime
import logging
import os
import warnings
from distutils.version import LooseVersion
from typing import Any
from typing import Optional
from typing import cast

import airflow
from airflow.configuration import conf
from airflow.models.xcom import BaseXCom
from airflow.utils.session import provide_session

try:
    from airflow.utils.session import NEW_SESSION
except ImportError:
    NEW_SESSION = None
try:
    from airflow.utils.helpers import exactly_one
    from airflow.utils.helpers import is_container
except ImportError:

    def is_container(obj: Any) -> bool:
        """Test if an object is a container (iterable) but not a string"""
        return hasattr(obj, '__iter__') and not isinstance(obj, str)

    def exactly_one(*args) -> bool:
        """
        Returns True if exactly one of *args is "truthy", and False otherwise.
        If user supplies an iterable, we raise ValueError and force them to unpack.
        """
        if is_container(args[0]):
            raise ValueError(
                "Not supported for iterable args. Use `*` to unpack your iterable "
                "in the function call."
            )
        return sum(map(bool, args)) == 1


import pendulum
from sqlalchemy.orm import Session
from sqlalchemy.orm import reconstructor
from sqlalchemy.orm.exc import NoResultFound

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
            raise RuntimeError(
                "Failed to find vineyard IPC socket configuration, "
                "please configure it using the environment variable "
                "$VINEYARD_IPC_SOCKET, or via airfow's vineyard.ipc_socket"
                "configuration."
            )
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
        Called by the ORM after the instance has been loaded from the DB or otherwise
        reconstituted i.e automatically deserialize Xcom value when loading from DB.
        """
        self.value = super().init_on_load()

    @classmethod
    @provide_session
    def _set(
        cls,
        key: str,
        value: Any,
        task_id: str,
        dag_id: str,
        execution_date: Optional[datetime.datetime] = None,
        session: Session = None,
        *,
        run_id: Optional[str] = None,
        map_index: int = -1,
    ) -> None:
        """:sphinx-autoapi-skip:"""
        # pylint: disable=import-outside-toplevel
        from airflow.models.dagrun import DagRun

        # pylint: enable=import-outside-toplevel

        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError(
                f"Exactly one of run_id or execution_date must be passed. "
                f"Passed execution_date={execution_date}, run_id={run_id}"
            )

        if run_id is None:
            message = (
                "Passing 'execution_date' to 'XCom.set()' is deprecated. "
                "Use 'run_id' instead."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            try:
                dag_run_id, run_id = (
                    session.query(DagRun.id, DagRun.run_id)
                    .filter(
                        DagRun.dag_id == dag_id, DagRun.execution_date == execution_date
                    )
                    .one()
                )
            except NoResultFound:
                raise ValueError(
                    f"DAG run not found on DAG {dag_id!r} at {execution_date}"
                ) from None
        else:
            dag_run_id = (
                session.query(DagRun.id)
                .filter_by(dag_id=dag_id, run_id=run_id)
                .scalar()
            )
            if dag_run_id is None:
                raise ValueError(
                    f"DAG run not found on DAG {dag_id!r} with ID {run_id!r}"
                )

        value = VineyardXCom.serialize_value(
            value=value,
            key=key,
            task_id=task_id,
            dag_id=dag_id,
            run_id=run_id,
            map_index=map_index,
        )

        # Remove duplicate XComs and insert a new one.
        if LooseVersion(airflow.__version__) < LooseVersion('2.3.0b1'):
            query = session.query(cls).filter(
                cls.key == key,
                cls.run_id == run_id,
                cls.task_id == task_id,
                cls.dag_id == dag_id,
            )
        else:
            query = session.query(cls).filter(
                cls.key == key,
                cls.run_id == run_id,
                cls.task_id == task_id,
                cls.dag_id == dag_id,
                cls.map_index == map_index,  # pylint: disable=no-member
            )

        # remove from vineyard
        targets = []
        for result in query.with_entities(VineyardXCom.value):
            targets.append(vineyard.ObjectID(BaseXCom.deserialize_value(result)))
        if targets:
            logger.info("Drop duplicates from vineyard: %s", targets)
            try:
                client = vineyard.connect(cls.options()['ipc_socket'])
                client.delete(targets)
            except Exception as e:  # pylint: disable=broad-except
                logger.error('Failed to drop duplicates from vineyard: %s', e)

        # remove from the underlying xcom db
        query.delete()
        session.commit()

        if LooseVersion(airflow.__version__) < LooseVersion('2.3.0b1'):
            # Work around Mypy complaining model not defining '__init__'.
            new = cast(Any, cls)(  # pylint: disable=unexpected-keyword-arg
                dag_run_id=dag_run_id,
                key=key,
                value=value,
                run_id=run_id,
                task_id=task_id,
                dag_id=dag_id,
            )
        else:
            # Work around Mypy complaining model not defining '__init__'.
            new = cast(Any, cls)(  # pylint: disable=unexpected-keyword-arg
                dag_run_id=dag_run_id,
                key=key,
                value=value,
                run_id=run_id,
                task_id=task_id,
                dag_id=dag_id,
                map_index=map_index,
            )
        session.add(new)
        session.flush()

    if LooseVersion(airflow.__version__) < LooseVersion('2.2.0b1'):

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
            query = session.query(cls).filter(
                cls.key == key,
                cls.execution_date == execution_date,
                cls.task_id == task_id,
                cls.dag_id == dag_id,
            )
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
            session.add(
                VineyardXCom(
                    key=key,
                    value=value,
                    execution_date=execution_date,
                    task_id=task_id,
                    dag_id=dag_id,
                )
            )
            session.commit()

    elif LooseVersion(airflow.__version__) < LooseVersion('2.3.0b1'):

        @classmethod
        @provide_session
        def set(
            cls,
            key: str,
            value: Any,
            task_id: str,
            dag_id: str,
            execution_date: Optional[datetime.datetime] = None,
            session: Session = None,
            *,
            run_id: Optional[str] = None,
        ) -> None:
            return cls._set(
                key=key,
                value=value,
                task_id=task_id,
                dag_id=dag_id,
                execution_date=execution_date,
                session=session,
                run_id=run_id,
            )

    else:

        @classmethod
        @provide_session
        def set(
            cls,
            key: str,
            value: Any,
            task_id: str,
            dag_id: str,
            execution_date: Optional[datetime.datetime] = None,
            session: Session = None,
            *,
            run_id: Optional[str] = None,
            map_index: int = -1,
        ) -> None:
            return cls._set(
                key=key,
                value=value,
                task_id=task_id,
                dag_id=dag_id,
                execution_date=execution_date,
                session=session,
                run_id=run_id,
                map_index=map_index,
            )

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
    def _clear(
        cls,
        execution_date: Optional[pendulum.DateTime] = None,
        dag_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session: Session = None,
        *,
        run_id: Optional[str] = None,
        map_index: Optional[int] = None,
    ) -> None:
        """:sphinx-autoapi-skip:"""
        from airflow.models import DagRun

        # Given the historic order of this function (execution_date was first argument)
        # to add a new optional param we need to add default values for everything :(
        if dag_id is None:
            raise TypeError("clear() missing required argument: dag_id")
        if task_id is None:
            raise TypeError("clear() missing required argument: task_id")

        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError(
                f"Exactly one of run_id or execution_date must be passed. "
                f"Passed execution_date={execution_date}, run_id={run_id}"
            )

        if execution_date is not None:
            message = (
                "Passing 'execution_date' to 'XCom.clear()' is deprecated. "
                "Use 'run_id' instead."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            run_id = (
                session.query(DagRun.run_id)
                .filter(
                    DagRun.dag_id == dag_id, DagRun.execution_date == execution_date
                )
                .scalar()
            )

        query = session.query(cls).filter_by(
            dag_id=dag_id, task_id=task_id, run_id=run_id
        )
        if map_index is not None:
            query = query.filter_by(map_index=map_index)

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

    if LooseVersion(airflow.__version__) < LooseVersion('2.2.0b1'):

        @classmethod
        @provide_session
        def clear(
            cls,
            execution_date: Optional[pendulum.DateTime] = None,
            dag_id: Optional[str] = None,
            task_id: Optional[str] = None,
            session: Session = None,
        ) -> None:
            return cls._clear(execution_date, dag_id, task_id, session)

    elif LooseVersion(airflow.__version__) < LooseVersion('2.3.0b1'):

        @classmethod
        @provide_session
        def clear(
            cls,
            execution_date: Optional[pendulum.DateTime] = None,
            dag_id: Optional[str] = None,
            task_id: Optional[str] = None,
            run_id: Optional[str] = None,
            session: Session = NEW_SESSION,
        ) -> None:
            return cls._clear(execution_date, dag_id, task_id, session, run_id=run_id)

    else:

        @classmethod
        @provide_session
        def clear(
            cls,
            execution_date: Optional[pendulum.DateTime] = None,
            dag_id: Optional[str] = None,
            task_id: Optional[str] = None,
            session: Session = NEW_SESSION,
            *,
            run_id: Optional[str] = None,
            map_index: Optional[int] = None,
        ) -> None:
            return cls._clear(
                execution_date,
                dag_id,
                task_id,
                session,
                run_id=run_id,
                map_index=map_index,
            )

    @staticmethod
    def serialize_value(
        value: Any,
        *,
        key: Optional[str] = None,
        task_id: Optional[str] = None,
        dag_id: Optional[str] = None,
        run_id: Optional[str] = None,
        map_index: Optional[int] = None,
    ):
        client = vineyard.connect(VineyardXCom.options()['ipc_socket'])
        value_id = client.put(value)
        if VineyardXCom.options()['persist']:
            client.persist(value_id)
        logger.debug("serialize_value: %s -> %r", value, value_id)
        return BaseXCom.serialize_value(repr(value_id))

    @staticmethod
    def deserialize_value(result: "VineyardXCom") -> Any:
        value = BaseXCom.deserialize_value(result)
        vineyard_value = VineyardXCom.post_resolve_value(result, value)
        logger.debug(
            "deserialize_value: %s ->  %s -> %s", result, value, vineyard_value
        )
        return vineyard_value

    @staticmethod
    @provide_session
    def post_resolve_value(
        result: "VineyardXCom", value: Any, session: Session = None
    ) -> Any:
        """The :code:`post_resolve_value` runs before the return the value to the
        operators to prepare necessary input data for the task.

        The post resolution will fill-up the occurrence if remote objects by
        of :code:`VineyardObjectRef` with the actual (remote) value by triggering
        a migration.

        It will also record the migrated xcom value into the db as well to make
        sure it can be dropped properly.
        """
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
