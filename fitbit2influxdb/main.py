import logging
from fitbit2influx import *

if __name__ == '__main__':
    logging.info('Get Started Fitbit2InfluxDB')
    fitbit_manager = Fitbit2InfluxdbManager()

    sched = BlockingScheduler()

    fitbit_a = fitbit_manager.addFitbit('fitbit_a')
    fitbit_b = fitbit_manager.addFitbit('fitbit_b')
    fitbit_c = fitbit_manager.addFitbit('fitbit_c')
    fitbit_d = fitbit_manager.addFitbit('fitbit_d')

    sched.start()