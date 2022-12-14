#!/usr/bin/env python
from numbers import Number
import os
import json
import logging
import time
import fitbit
import datetime
import get2token
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
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
]

FITBIT_ACTIVITES_NON_INTRADAY_RESOURCE = [
    'activities-activityCalories',
    'activities-caloriesBMR',
    'activities-tracker-activityCalories',
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

FITBIT_BRATTARY_RESOURCE = 'devices-battery'

# fitbit resource에 대응되는 단위
FITBIT_RESOURCE_TO_MEASUREMENT = {
    "activities-activityCalories": "cal",
    "activities-calories": "cal",
    "activities-caloriesBMR":"cal",
    "activities-distance":"mile",
    "activities-elevation":"ft",
    "activities-floors":"floors",
    "activities-heart":"bpm",
    "activities-minutesFairlyActive":"min",
    "activities-minutesLightlyActive":"min",
    "activities-minutesSedentary":"min",
    "activities-minutesVeryActive":"min",
    "activities-steps":"steps",
    "activities-tracker-activityCalories":"cal",
    "activities-tracker-calories":"cal",
    "activities-tracker-distance":"mile",
    "activities-tracker-elevation":"ft",
    "activities-tracker-floors":"floors",
    "activities-tracker-minutesFairlyActive":"min",
    "activities-tracker-minutesLightlyActive":"min",
    "activities-tracker-minutesSedentary":"min",
    "activities-tracker-minutesVeryActive":"min",
    "activities-tracker-steps":"steps",
    "body-bmi":"BMI",
    "body-fat":"%",
    "body-weight":"lb",
    "sleep-awakeningsCount":"times awaken",
    "sleep-efficiency":"%",
    "sleep-minutesAfterWakeup":"min",
    "sleep-minutesAsleep":"min",
    "sleep-minutesAwake":"min",
    "sleep-minutesToFallAsleep":"min",
    "sleep-startTime":"startTime",
    "sleep-timeInBed":"min"
}

# influxdb time에 대한 foramt
DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f+09:00'

EXAM_FITBIT_FILE = {
    "access_token": "DEFAULT_ACCESS_TOKEN",
    "refresh_token": "DEFAULT_REFRESH_TOKEN",
    "client_id": "DEFAULT_CLIENT_ID",
    "client_secret": "DEFAULT_CLIENT_SECRET",
    "last_saved_at": "DEFAULT_last_saved_at",
}

class Fitbit2Influxdb:
    def __init__(self, entity_id):
        self.entity_id = entity_id
        self.access_token = ''
        self.refresh_token = ''
        self.client_id = ''
        self_client_secret: str | None = ''
        self.last_saved_at: int | None = 0
        self.authd_client = None
        self.influxdb_client = None

        self.fitbit_load_json()
        self.connect_fitbit_api()
        self.fitbit_save_json()
        self.connect_influxdb()


        # print(self.authd_client.get_devices())

        self.sched = BackgroundScheduler()

        self.sched.add_job(self.update, 'cron', minute='29', id=self.entity_id+'-update-1')
        self.sched.add_job(self.update, 'cron', minute='58', id=self.entity_id+'-update-2')

        self.sched.start()

    def connect_influxdb(self):
        ip, port, username, password, database = self.influxdb_load_env()
        if port.isdigit():
            port = int(port)
        else:
            logging.error(self.entity_id + "- Port is not number. Please check")
            raise TypeError('Port is not number. Please check')
        self.influxdb_client = InfluxDBClient(ip, port, username, password, database)
        logging.info(self.entity_id + '- Connected influxDB')
    
    # fitbit API에 토근값과 함께 접속
    def connect_fitbit_api(self): 
        # server = get2token.OAuth2Server(self.client_id, self.client_secret)
        self.authd_client = fitbit.Fitbit(
            self.client_id, 
            self.client_secret, 
            access_token=self.access_token, 
            refresh_token=self.refresh_token,
            redirect_uri='http://127.0.0.1:8080/',
            refresh_cb=lambda x: None
        )
        # server.browser_authorize()

        # self.authd_client = server.fitbit

        logging.info(self.entity_id + '- Connected fitbit api')
    
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
            logging.error(self.entity_id + "- Can't load environment. Please check .env files")
            raise Exception(self.entity_id + '- Check .env files. you should fill IP, PORT, USERNAME, PASSWORD, DATABASE.')
        
        logging.info(self.entity_id + "- Success load influxdb infomation.")
        return ip, port, username, password, database

    def fitbit_load_json(self):
        if os.path.exists(self.entity_id + '.json'):
            with open(self.entity_id + '.json', 'r') as f:
                load_data = json.load(f)
                self.last_saved_at = load_data['last_saved_at']
                self.access_token = load_data['access_token']
                self.refresh_token = load_data['refresh_token']
                self.client_id = load_data['client_id']
                self.client_secret = load_data['client_secret']
                logging.info(self.entity_id + '- Finish load json file')

                # DEFAULT check
                
        else:
            with open(self.entity_id + '.json', 'w') as f:
                json.dump(EXAM_FITBIT_FILE, f, indent=4)
            logging.error(self.entity_id + '- Cannot access' + self.entity_id + '.json: No such file and directory.')
            raise Exception('Cannot access' + self.entity_id + '.json: No such file and directory. Please check README')
    
    def fitbit_save_json(self):
        token = self.authd_client.client.session.token
        config_contents = {
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "last_saved_at": int(time.time()),
        }

        with open(self.entity_id + '.json', 'w') as json_file:
            json.dump(config_contents, json_file, indent=4)
        
        logging.info(self.entity_id + '- Finish save json file. filename : ' + self.entity_id + '.json')

    def safe_transform(self, value):
        result:float | str | None = None

        try:
            result = float(value)
        except:
            result = value

        return result

    def influxdb_write_intraday_data(self, resource, dataset):
        for data in dataset:
            time = data['time']
            value = data['value']

            timestamp = self.fitbit_time_to_datetime(time).strftime(DATETIME_FORMAT)

            payload_json = {
                'measurement': FITBIT_RESOURCE_TO_MEASUREMENT[resource],
                'tags': {
                    'domain': 'sensor',
                    'entity_id': self.entity_id + '_' + resource,
                },
                'fields': {
                    'value': self.safe_transform(value)
                },
                'time': timestamp
            }
            try:
                self.influxdb_client.write_points([payload_json])
            except Exception as e:
                logging.error(self.entity_id + ' - error on influxdb_write_non_intraday_date.' + e.__str__())

        logging.info(self.entity_id + '- ' + resource + '- Success write data')

            
    def influxdb_write_non_intraday_date(self, resource, value):
        payload_json = {
            'measurement': FITBIT_RESOURCE_TO_MEASUREMENT[resource],
            'tags': {
                'domain': 'sensor',
                'entity_id': self.entity_id + '_' + resource,
            },
            'fields': {
                'value': self.safe_transform(value)
            },
        }

        try:
            self.influxdb_client.write_points([payload_json])
        except Exception as e:
            logging.error(self.entity_id + ' - error on influxdb_write_non_intraday_date.' + e.__str__())
        
        logging.info(self.entity_id + '- ' + resource + '- Success write data')

    def update(self):
        if int(time.time()) - self.last_saved_at >= 3600:
            self.authd_client.client.refresh_token()
            self.fitbit_save_json()
            logging.info(self.entity_id + '- Before expires refresh token! refreshing token...')

        response_intraday_result = []
        response_non_intraday_result = []

        for resource in FITBIT_ACTIVITES_INTRADAY_RESOURCE:
            detail_level = '1sec' if resource == 'activities-heart' else '1min'
            response = self.authd_client.intraday_time_series(resource.replace('-', '/'), detail_level=detail_level)

            dataset = response[resource + '-intraday']['dataset']

            response_intraday_result.append({'resource': resource, 'dataset': dataset})

            logging.info(self.entity_id + '- ' + resource + '- resource append done!')
        
        for resource in (FITBIT_ACTIVITES_NON_INTRADAY_RESOURCE + FITBIT_BODY_RESOURCE + FITBIT_SLEEP_RESOURCE):
            response = self.authd_client.intraday_time_series(resource.replace('-', '/'), detail_level='1min')

            data = response[resource][0]['value']

            if data == '':
                data = '-'

            response_non_intraday_result.append({'resource': resource, 'data': data})

            logging.info(self.entity_id + '- ' + resource + '- resource append done!')
        
        for item in response_intraday_result:
            self.influxdb_write_intraday_data(item['resource'], item['dataset'])
        
        for item in response_non_intraday_result:
            self.influxdb_write_non_intraday_date(item['resource'], item['data'])
        
        logging.info(self.entity_id + '- update done.')

class Fitbit2InfluxdbManager:
    def __init__(self):
        self.fitbit2influxdb_list = []
    
    def addFitbit(self, entity_name: str):
        for fitbit2influxdb in self.fitbit2influxdb_list:
            if fitbit2influxdb.entity_id == entity_name:
                raise Exception('Same Entity_id. Please don\'t overlap name')
        
        new_fitbit = Fitbit2Influxdb(entity_name)
        self.fitbit2influxdb_list.append(new_fitbit)
        return new_fitbit