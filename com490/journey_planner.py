
import os
import json
import time
import re
import trino
import warnings
import pandas as pd
import base64 as b64
from typing import List
from contextlib import closing
from trino.dbapi import connect
from urllib.parse import urlparse
from db.query_runner import execute, fetch
from routing.time_utils import time_to_seconds
from trino.auth import BasicAuthentication, JWTAuthentication
from routing.router import Router
from db.sql_paths import (
    CREATE_STOPS, 
    CREATE_STOP_TIMES, 
    CREATE_STOP_TO_STOP, 
    CREATE_WALK_PATHS,
)




class JourneyPlanner:
    def __init__(self, schema ):
        self.userns = schema
        self.sharedns = 'iceberg.com490_iceberg'
        trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
        trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
        self.conn = connect(
            host=trinoUrl.hostname,
            port=trinoUrl.port,
            auth=trinoAuth,
            http_scheme=trinoUrl.scheme,
            verify=True
        )
        self._router  = Router(self.conn, self.userns, self.sharedns)
        self.id_to_name = None
        self.name_to_id = None

    def prepare(self, regions=List[str]):
        region_uuids_sql = ",".join([f"'{u}'" for u in regions])

        max_pub_date = self._get_max_pub_date()
        print('Max publication date:',max_pub_date)

        execute(self.conn, CREATE_STOPS,
            userns=self.userns,
            sharedns=self.sharedns,
            region_uuids_sql=region_uuids_sql,
            max_pub_date=max_pub_date,
        )
        print('Finished Creating stops..')

        execute(self.conn, CREATE_STOP_TIMES,
            userns=self.userns,
            sharedns=self.sharedns,
            max_pub_date=max_pub_date,
        )
        print('Finished Creating stops times..')

        execute(self.conn, CREATE_STOP_TO_STOP,
            userns=self.userns,
        )
        execute(self.conn, CREATE_WALK_PATHS,
            userns=self.userns,
        )

        print('Finished Creating walk stat times..')

        stops_df = pd.read_sql(f"SELECT * FROM {self.userns}.stops", self.conn)
        self.name_to_id = dict(zip(stops_df['stop_name'], stops_df['stop_id']))
        self.id_to_name = {v: k for k, v in self.name_to_id.items()}
            
    def route(self, start_id, end_id, departs:str, day:str, max_walk_m:int, top_k:int = 1, transfer_buffer:int = 60, static=True):
        if static:
            route_ = self._router.route(
                start_id=start_id,
                end_id=end_id,
                departs=departs,
                day=day,
                max_walk_m=max_walk_m,
                transfer_buffer_sec=transfer_buffer
            )
        else:
            dist, prev, route_, results_df =  self._router.route_dynamic(
                start_id=start_id,
                end_id=end_id,
                departs=departs,
                day=day,
                max_walk_m=max_walk_m,
                top_k=top_k,
                transfer_buffer_sec=transfer_buffer,
                id_to_name=self.id_to_name
            )
        return route_

    ''' old
    def route_k(self, start_id, end_id, departs: str, day: str, max_walk_m: int, k: int = 10):
        return self._router.route_dynamic_k(
            start_id=start_id,
            end_id=end_id,
            departs=departs,
            day=day,
            max_walk_m=max_walk_m,
            k=k,
        )
    '''

    ''' old
    def route_latest(self, start_id, end_id, deadline, day, max_walk_m, slack_steps=None):
        return self._router.route_dynamic_backward(
            start_id=start_id, end_id=end_id,
            deadline=deadline, day=day,
            max_walk_m=max_walk_m,
            top_k=top_k,
            transfer_buffer=transfer_buffer,
            slack_steps=slack_steps or (0, 180, 360, 540, 720),
        )
    '''

    def route_latest(self, start_id, end_id, deadline, day, max_walk_m, top_k = 1, slack_steps=None, buffer_steps=None):
        return self._router.route_dynamic_backward(
            start_id=start_id, end_id=end_id,
            deadline=deadline, day=day,
            max_walk_m=max_walk_m,
            top_k=top_k,
            slack_steps=slack_steps or (0, 180, 360, 540, 720),
            buffer_steps=buffer_steps or (60,),
        )
    
    def _get_max_pub_date(self):
        max_pub_date = pd.read_sql(f"""
            SELECT MAX(pub_date) AS max_pub_date
            FROM {self.sharedns}.sbb_stops
            """, self.conn).iloc[0, 0]
        return max_pub_date
