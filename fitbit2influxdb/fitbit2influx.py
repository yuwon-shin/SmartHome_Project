#!/usr/bin/env python
import os
import json
import logging
import time
import fitbit

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

FITBIT_RESOURCE = ['activities/activityCalories', 
'activities/calories', 
'activities/caloriesBMR',
'activities/distance',
'activities/elevation',
'activities/floors',
'activities/heart',
'activities/minutesFairlyActive',
'activities/minutesLightlyActive',
'activities/minutesSedentary',
'activities/minutesVeryActive',
'activities/steps',
'activities/tracker/activityCalories',
'activities/tracker/calories',
'activities/tracker/distance',
'activities/tracker/elevation',
'activities/tracker/floors',
'activities/tracker/minutesFairlyActive',
'activities/tracker/minutesLightlyActive',
'activities/tracker/minutesSedentary',
'activities/tracker/minutesVeryActive',
'activities/tracker/steps',
'body/bmi',
'body/fat',
'body/weight',
'sleep/awakeningsCount',
'sleep/efficiency',
'sleep/minutesAfterWakeup',
'sleep/minutesAsleep',
'sleep/minutesAwake',
'sleep/minutesToFallAsleep',
'sleep/startTime',
'sleep/timeInBed', ]

FITBIT_ACTIVITES_INTRADAY_RESOURCE = []

FITBIT_ACTIVITES_NON_INTRADAY_RESOURCE = []

FITBIT_SLEEP_RESOURCE = []

FITBIT_BODY_RESOURCE = []

class Fitbit2Influxdb:
    def __init__(self):
        self.access_token = ''
        self.refresh_token = ''
        self.client_id = ''
        self_client_secret: str | None = ''
        self.expires_at: int | None = 0
        self.authd_client = None

        self.load_json()
        self.connect_fitbit_api()
        self.save_json()
    
    def connect_fitbit_api(self):
        self.authd_client = fitbit.Fitbit(self.client_id, self.client_secret, access_token=self.access_token, refresh_token=self.refresh_token)
        logging.info('Connected fitbit api')

        if int(time.time()) - self.expires_at > 3600:
            self.authd_client.client.refresh_token()
            logging.info('Expires refresh token! refreshing token...')

    def load_json(self):
        if os.path.exists('save.json'):
            with open('save.json', 'r') as f:
                load_data = json.load(f)
                if 'LAST_SAVED_AT' in load_data:
                    self.expires_at = load_data['LAST_SAVED_AT']
                    # if (last_saved_at + 3600) <= int(time.time()):
                    #     logging.error('Expires refresh token: Please reauthcation and modify token value')
                    #     raise Exception('Expires refresh token: Please reauthcation and modify token value')
                self.access_token = load_data['access_token']
                self.refresh_token = load_data['refresh_token']
                self.client_id = load_data['client_id']
                self.client_secret = load_data['client_secret']
                logging.info('Finish load json file')
        else:
            logging.error('Cannot access "save.json": No such file and directory.')
            raise Exception('Cannot access "save.json": No such file and directory. Please check README')
    
    def save_json(self):
        token = self.authd_client.client.session.token
        config_contents = {
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "LAST_SAVED_AT": int(time.time()),
        }

        with open('save.json', 'w') as json_file:
            json.dump(config_contents, json_file, indent=4)
        
        logging.info('Finish save json file')

    def update(self):
        result_response = {}
        for resource in FITBIT_RESOURCE:
            detail_level = '1sec' if resource == 'activities/heart' else '1min'
            response = self.authd_client.intraday_time_series(resource, detail_level=detail_level)
            result_response[resource] = response
        

        for resource_response in result_response:
            file_name = resource_response.replace('/', '-') + '.json'
            with open(file_name, 'w') as json_file:
                json.dump(result_response[resource_response], json_file)

if __name__ == '__main__':
    logging.info('Get Started Fitbit2InfluxDB')
    test = Fitbit2Influxdb()

    test.update()
