import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import re
import sys
import numpy as np

import signal
import tornado.web
import tornado.ioloop
from tornado.options import options
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import json
import os
import traceback
import imp
import time

iswindows = True
'''
PATH_PRE='/home/app/algorithm_web/'
LOG_PATH='../logs/'
sys.path.append(PATH_PRE+'web/xmodule/')
sys.path.append(PATH_PRE+'web/log/')
scriptPath = PATH_PRE+'web/customize/'
sys.path.append(scriptPath)
'''

PATH_PRE = 'E:\\project\\2021_Project\\TornadoWeb\\'
LOG_PATH = PATH_PRE + 'web\\logs\\'
sys.path.append(PATH_PRE + 'web\\xmodule\\')
sys.path.append(PATH_PRE + 'web\\log\\')
scriptPath = PATH_PRE + 'web\\customize\\'
sys.path.append(scriptPath)
import dmc, pid

MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 3
tornado.options.define("port", default=9000, type=int, multiple=False, help="this s is a port")


def log_init():
    log_fmt = '%(asctime)s\\tFile \\"%(filename)s\\",line %(lineno)s\\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler = TimedRotatingFileHandler(filename=LOG_PATH + "thread_.log", when="D", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    log_file_handler.setFormatter(formatter)
    log_file_handler.setLevel(logging.DEBUG)
    log = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # datefmt='%a, %d %b %Y %H:%M:%S',
                        # filename='/home/URL/client/test_log.log',
                        filemode='a')
    log.addHandler(log_file_handler)


def load(module_name, module_path):
    fp, pathname, description = imp.find_module(module_name, [module_path])
    try:
        return imp.load_module(module_name, fp, pathname, description)
    finally:
        if fp:
            fp.close()


def narrayConvert(value):
    return value.tolist() if type(value) == np.ndarray else value


def listConvert(value):
    return np.array(value) if type(value) == list else value


class DmcHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)

    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        # self.set_header('Access-Control-Allow-Origin', 'http://localhost:8080')
        # self.set_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    @run_on_executor
    def contrl_dmc(self, requestdata):
        print('dmc pid=', os.getpid())
        try:
            input_data = requestdata['input']
            context = requestdata['context']
            data = dmc.main(input_data, context)
            resp = {'data': data, 'algorithmContext': context, 'message': '', 'status': 200}
            return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            return error

    @tornado.gen.coroutine
    def post(self):
        data = json.loads(self.request.body)
        result = yield self.contrl_dmc(data)
        self.write(result)
        sys.stdout.write("pid=%s,request ip =%s,serve by =%s result_mv=%s,result_dmv=%s\n" % (
            str(os.getpid()), str(self.request.remote_ip), str(options.port),
            str(result['mv'] if 'mv' in result else ''),
            str(result['dmv'] if 'dmv' in result else '')))
        sys.stdout.flush()
        pass


class PidHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)

    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        # self.set_header('Access-Control-Allow-Origin', 'http://localhost:8080')
        # self.set_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    @run_on_executor
    def contrl_pid(self, requestdata):
        try:
            input_data = requestdata['input']
            context = requestdata['context']
            data = pid.main(input_data, context)
            resp = {'data': data, 'algorithmContext': context, 'message': '', 'status': 200}
            return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            return error

    @tornado.gen.coroutine
    def post(self):
        data = json.loads(self.request.body)
        result = yield self.contrl_pid(data)
        self.write(result)
        sys.stdout.write("request ip =%s,serve by =%s\n" % (str(self.request.remote_ip), str(options.port)))
        sys.stdout.flush()


class CustomizeHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)

    def initialize(self):
        self.set_default_header()

    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        # self.set_header('Access-Control-Allow-Origin', 'http://localhost:8080')
        # self.set_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    @run_on_executor
    def contrl_customize(self, requestdata):
        try:
            modelId = requestdata['modelId']
            input_data = requestdata['input']
            context = requestdata['context']
            scriptcontext = requestdata['pythoncontext']
            with open(scriptPath + "%s.py" % modelId, 'w+', encoding='utf-8') as f:
                f.write(scriptcontext)
            do = load(str(modelId), scriptPath)
            result = do.main(input_data, context)
            for key, value in context.items():
                context[key] = narrayConvert(value)
            resp = {'data': result, 'algorithmContext': context, 'message': '', 'status': 200}
            sys.stdout.write("request ip =%s,serve by =%s,modelId=%s\n" % (
                str(self.request.remote_ip), str(options.port), str(modelId)))
            return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            return error

    @tornado.gen.coroutine
    def post(self):
        data = json.loads(self.request.body)
        result = yield self.contrl_customize(data)
        self.write(result)
        sys.stdout.write(
            "pid=%s,request ip =%s,serve by =%s\n" % (str(os.getpid()), str(self.request.remote_ip), str(options.port)))
        sys.stdout.flush()


class ProxyServer:

    def __init__(self):
        self.__app = None
        self.__server = None
        self.ioloop = tornado.ioloop.IOLoop.instance()

    def run(self):
        tornado.options.parse_command_line()
        self.__app = tornado.web.Application([
            (r'/dmc', DmcHandler),
            (r'/pid', PidHandler),
            (r'/customize', CustomizeHandler)
        ], debug=False, autoreload=False)
        self.__server = tornado.httpserver.HTTPServer(self.__app, xheaders=True)
        sys.stdout.write("bind port=%s\n" % (str(tornado.options.options.port)))
        sys.stdout.flush()
        self.__server.bind(tornado.options.options.port, '0.0.0.0')
        self.__server.start() if iswindows else self.__server.start(0)
        self.ioloop.start()

    def sig_handler(self, sig, frame):
        logging.warning('Caught signal: %s', sig)
        self.ioloop.add_callback_from_signal(self.shutdown)

    def shutdown(self):
        logging.info('Stopping http server')
        self.__server.stop()
        logging.info('Will shutdown in %s seconds ...', MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)
        deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

        def stop_loop():
            now = time.time()
            if now < deadline:
                self.ioloop.add_timeout(now + 1, stop_loop)
            else:
                self.ioloop.stop()
                logging.info('Tornado Shutdown')

        stop_loop()

    @staticmethod
    def stop():
        pid = os.getpid()
        print("stop pid=%d" % pid)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGINT)


if __name__ == '__main__':
    log_init()
    serve = ProxyServer()
    signal.signal(signal.SIGTERM, serve.sig_handler)
    signal.signal(signal.SIGINT, serve.sig_handler)
    serve.run()
