#!/usr/bin/env python
import os
import json
import logging
import time
import fitbit
import datetime
from influxdb import InfluxDBClient
from dotenv import load_dotenv

# .env 파일을 불러와 환경변수에 넣어줌.
load_dotenv()

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log를 파일에 출력
file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Fitbit이 제공하는 Resource
FITBIT_ACTIVITES_INTRADAY_RESOURCE = [
    'activities-calories',
    'activities-distance',
    'activities-elevation',
    'activities-floors',
    'activities-heart',
    'activities-minutesFairlyActive',
    'activities-minutesLightlyActive',
    'activities-minutesSedentary',
    'activities-minutesVeryActive',
    'activities-steps',
    'activities-tracker-activityCalories',
]

FITBIT_ACTIVITES_NON_INTRADAY_RESOURCE = [
    'activities-activityCalories',
    'activities-caloriesBMR',
    'activities-tracker-calories',
    'activities-tracker-distance',
    'activities-tracker-elevation',
    'activities-tracker-floors',
    'activities-tracker-minutesFairlyActive',
    'activities-tracker-minutesLightlyActive',
    'activities-tracker-minutesSedentary',
    'activities-tracker-minutesVeryActive',
    'activities-tracker-steps',
]

FITBIT_SLEEP_RESOURCE = [
    'sleep-awakeningsCount',
    'sleep-efficiency',
    'sleep-minutesAfterWakeup',
    'sleep-minutesAsleep',
    'sleep-minutesAwake',
    'sleep-minutesToFallAsleep',
    'sleep-startTime',
    'sleep-timeInBed'
]

FITBIT_BODY_RESOURCE = [
    'body-weight',
    'body-fat',
    'body-bmi'
]

FITBIT_BRATTARY_RESOURCE = 'device-bettray'

DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

EXAM_FITBIT_FILE = {
    "access_token": "DEFAULT_ACCESS_TOKEN",
    "refresh_token": "DEFAULT_REFRESH_TOKEN",
    "client_id": "DEFAULT_CLIENT_ID",
    "client_secret": "DEFAULT_CLIENT_SECRET",
    "EXPIRES_TIME": "DEFAULT_EXPIRES_TIME",
}

class Fitbit2Influxdb:
    def __init__(self, entity_id='fitbit'):
        self.entity_id = entity_id
        self.access_token = ''
        self.refresh_token = ''
        self.client_id = ''
        self_client_secret: str | None = ''
        self.expires_at: int | None = 0
        self.authd_client = None
        self.influxdb_client = None

        self.fitbit_load_json()
        self.connect_fitbit_api()
        self.fitbit_save_json()
        self.connect_influxdb()

    def connect_influxdb(self):
        ip, port, username, password, database = self.influxdb_load_env()
        if port.isdigit():
            port = int(port)
        else:
            logging.error("Port is not number. Please check")
            raise TypeError('Port is not number. Please check')
        self.influxdb_client = InfluxDBClient(ip, port, username, password, database)
        logging.info('Connected influxDB')
    
    # fitbit API에 토근값과 함께 접속
    def connect_fitbit_api(self): 
        self.authd_client = fitbit.Fitbit(self.client_id, self.client_secret, access_token=self.access_token, refresh_token=self.refresh_token)
        logging.info('Connected fitbit api')

        if int(time.time()) - self.expires_at > 3600:
            self.authd_client.client.refresh_token()
            logging.info('Before expires refresh token! refreshing token...')
    
    def fitbit_time_to_datetime(self, time):
        today_date = datetime.date.today()
        time_date = datetime.time.fromisoformat(time)

        result = datetime.datetime.combine(today_date, time_date)

        return result

    def influxdb_load_env(self):
        ip = os.environ['IP']
        port = os.environ['PORT']
        username = os.environ['USERNAME']
        password = os.environ['PASSWORD']
        database = os.environ['DATABASE']

        if not(ip and port and username and password and database):
            logging.error("Can't load environment. Please check .env files")
            raise Exception('Check .env files. you should fill IP, PORT, USERNAME, PASSWORD, DATABASE.')
        
        logging.info("Success load influxdb infomation.")
        return ip, port, username, password, database

    def fitbit_load_json(self):
        if os.path.exists(self.entity_id + '.json'):
            with open(self.entity_id + '.json', 'r') as f:
                load_data = json.load(f)
                self.expires_at = load_data['EXPIRES_TIME']
                self.access_token = load_data['access_token']
                self.refresh_token = load_data['refresh_token']
                self.client_id = load_data['client_id']
                self.client_secret = load_data['client_secret']
                logging.info('Finish load json file')
        else:
            with open(self.entity_id + '.json', 'w') as f:
                json.dump(EXAM_FITBIT_FILE, f, indent=4)
            logging.error('Cannot access' + self.entity_id + '.json: No such file and directory.')
            raise Exception('Cannot access' + self.entity_id + '.json: No such file and directory. Please check README')
    
    def fitbit_save_json(self):
        token = self.authd_client.client.session.token
        config_contents = {
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "LAST_SAVED_AT": int(time.time()),
        }

        with open(self.entity_id + '.json', 'w') as json_file:
            json.dump(config_contents, json_file, indent=4)
        
        logging.info('Finish save json file. filename : ' + self.entity_id + '.json')

    def influxdb_write_intraday_data(self, resource, dataset):
        for data in dataset:
            time = data['time']
            value = data['value']

            timestamp = self.fitbit_time_to_datetime(time).strftime(DATETIME_FORMAT)

            payload_json = {
                'measurement': 'abc',
                'tags': {
                    'domain': 'sensor',
                    'entity_id': self.entity_id,
                },
                'fields': {
                    'value': value
                },
                'timestamp': timestamp
            }

            

    def influxdb_write_non_intraday_date(self, resource, value):
        pass

    def update(self):
        for resource in FITBIT_ACTIVITES_INTRADAY_RESOURCE:
            detail_level = '1sec' if resource == 'activities-heart' else '1min'
            response = self.authd_client.intraday_time_series(resource.replace('-', '/'), detail_level=detail_level)

            dataset = response[resource + '-intraday'][dataset]

            self.influxdb_write_intraday_data(resource, dataset)
        

if __name__ == '__main__':
    logging.info('Get Started Fitbit2InfluxDB')
    test = Fitbit2Influxdb()

    # foramt_test = '%Y-%m-%dT%H:%M:%S.%fZ'

    # test_json = {
    #     'measurement': 'abc',
    #     'tags': {
    #         'domain': 'sensor',
    #         'entity_id': 'test',
    #     },
    #     'fields': {
    #         'value': 'value'
    #     },
    #     'timestamp': "2022-08-09T00:00:00.0000Z"

    # }
    # test.influxdb_client.write()

    result = test.authd_client.get_devices()

    print(result)
    # test.update()
