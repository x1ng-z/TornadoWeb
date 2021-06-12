import signal
import tornado.web
import tornado.ioloop
from tornado.options import options
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from tornado import gen, web, ioloop, template
from tornado.gen import coroutine, Future
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from multiprocessing import Process, Pipe
import json

import os
import traceback
import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import sys, imp
import numpy as  np
import gevent.pywsgi
import logging

'''
sys.path.append('/home/app/algorithm_web/web/xmodule/')
sys.path.append('/home/app/algorithm_web/web/log/')
scriptPath = '/home/app/algorithm_web/web/customize/'
sys.path.append(scriptPath)
'''
sys.path.append('E:\\project\\2021_Project\\TornadoWeb\web\\xmodule\\')
sys.path.append('E:\\project\\2021_Project\\TornadoWeb\web\\log\\')
scriptPath = 'E:\\project\\2021_Project\\TornadoWeb\web\\customize\\'
sys.path.append(scriptPath)


import logconfig

logconfig.log_init()
import time
import dmc, pid

tornado.options.define("port",default=9000,type=int,multiple=False,help="this s is a port")


import signal

global http_server

def sig_handler(sig, frame):
    """信号处理函数
    """
    logging.info("rev sig=%s" % (str(sig)))
    tornado.ioloop.IOLoop.instance().add_callback(shutdown)


def shutdown():
    """进程关闭处理
    """
    # 停止接受Client连接
    logging.info('Stopping http server')

    http_server.stop()

    io_loop = tornado.ioloop.IOLoop.instance()
    deadline = time.time() + 2  # 设置最长强制结束时间

    def stop_loop():
        now = time.time()
        if now < deadline:
            io_loop.add_timeout(now + 1, stop_loop)
        else:
            io_loop.stop()
            logging.info('Shutdown tornado')

    stop_loop()


def load(module_name, module_path):
    '''使用imp的两个函数find_module，load_module来实现动态调用Python脚本。如果发现异常，需要解除对文件的占用'''
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


def contrl_dmc( cnn, requestdata):
        print('dmc pid=', os.getpid())
        try:
            # requestdata = json.loads(request.body)
            input_data = requestdata['input']
            context = requestdata['context']
            data = dmc.main(input_data, context)
            resp = {'data': data, 'algorithmContext': context, 'message': '', 'status': 200}
            cnn.send(resp)
            cnn.close()
            # return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            # return error
            cnn.send(error)
            cnn.close()

class DmcHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)

    def initialize(self):
        # self.request.method = 'POST'
        # print(self.request.method, type(self.request.method))
        self.set_default_header()
    def set_default_header(self):
        # print("setting headers!!!")
        self.set_header('Access-Control-Allow-Origin', '*')
        # self.set_header('Access-Control-Allow-Origin', 'http://localhost:8080')
        # self.set_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    @run_on_executor
    def contrl_dmc(self,requestdata):
        print('dmc pid=', os.getpid())
        try:
            # requestdata = json.loads(request.body)
            input_data = requestdata['input']
            context = requestdata['context']
            data = dmc.main(input_data, context)
            resp = {'data': data, 'algorithmContext': context, 'message': '', 'status': 200}
            return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            return error
            # cnn.send(error)
            # cnn.close()

    # @web.asynchronous
    # @tornado.web.asynchronous
    # @tornado.gen.coroutine
    def test1(self,input_data):
        # print("time=%s data=%s" % (time.time(), self.request.body))
        data = json.loads(self.request.body)
        # 向响应中，添加数据
        # result = yield self.contrl_dmc(data)
        parent_conn, child_conn = Pipe()

        p = Process(target=self.contrl_dmc, args=(child_conn, data,))
        p.start()
        data = parent_conn.recv()
        # print("result=%s" % data)  # prints "[42, None, 'hello']"
        p.join()
        # self.contrl_dmc(data)
        self.write(data)

    def test2(self,input_data):
        print("time=%s data=%s" % (time.time(), input_data))
        data = json.loads(input_data)
        # 向响应中，添加数据
        # result = yield self.contrl_dmc(data)
        result = self.contrl_dmc(data)
        self.write(result)


    @tornado.gen.coroutine
    def post(self):
        # self.test2(self.request.body)
        # start=time.time()
        # print("time=%s " % (time.time(),))
        data = json.loads(self.request.body)
        result = yield self.contrl_dmc(data)
        self.write(result)
        sys.stdout.write("pid=%s,request ip =%s,serve by =%s result_mv=%s,result_dmv=%s\n" % (str( os.getpid()),str(self.request.remote_ip), str(options.port),str(result['mv'] if 'mv' in result else ''),str(result['dmv'] if 'dmv' in result else '')))
        sys.stdout.flush()
        pass


class PidHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)
    def initialize(self):
        # self.request.method = 'POST'
        # print(self.request.method, type(self.request.method))
        self.set_default_header()
    def set_default_header(self):
        # print("setting headers!!!")
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
            # requestdata = json.loads(request.body)
            input_data = requestdata['input']
            context = requestdata['context']
            data = pid.main(input_data, context)
            resp = {'data': data, 'algorithmContext': context, 'message': '', 'status': 200}
            return resp
        except Exception as e:
            logging.error('%s' % traceback.format_exc())
            error = {'message': '%s' % traceback.format_exc(), 'status': 123456}
            return error

    # @web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        data = json.loads(self.request.body)
        result = yield self.contrl_pid(data)
        self.write(result)
        sys.stdout.write( "request ip =%s,serve by =%s\n" % (str(self.request.remote_ip), str(options.port)))
        sys.stdout.flush()

class CustomizeHandler(tornado.web.RequestHandler):
    max_thread_num = 10
    executor = ThreadPoolExecutor(max_workers=max_thread_num)
    def initialize(self):
        # self.request.method = 'POST'
        # print(self.request.method, type(self.request.method))
        self.set_default_header()
    def set_default_header(self):
        # print("setting headers!!!")
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
            # requestdata = json.loads(request.body)

            modelId = requestdata['modelId']
            input_data = requestdata['input']
            context = requestdata['context']
            scriptcontext = requestdata['pythoncontext']
            # 文件存储
            with open(scriptPath + "%s.py" % modelId, 'w+',encoding='utf-8') as f:
                f.write(scriptcontext)

            do = load(str(modelId), scriptPath)
            result = do.main(input_data, context)
            for key, value in context.items():
                context[key] = narrayConvert(value)
            resp = {'data': result, 'algorithmContext': context, 'message': '', 'status': 200}
            sys.stdout.write("request ip =%s,serve by =%s,modelId=%s\n" %(str(self.request.remote_ip),str(options.port),str(modelId)))
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
        sys.stdout.write("pid=%s,request ip =%s,serve by =%s\n" % (str( os.getpid()),str(self.request.remote_ip), str(options.port)))
        sys.stdout.flush()

def test():
    ppid = os.getppid()
    pid = os.getpid()
    print("test ppid=%d,pid=%d" % (ppid,pid))
    time.sleep(5)
    print("try to kill %d" % pid)
    os.kill(pid, signal.SIGTERM)

if __name__ == '__main__':
    # 等待supervisor发送进程结束信号
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    # signal.signal(signal.SIGKILL, sig_handler)
    # signal.signal(signal., sig_handler)
    tornado.options.parse_command_line()
    sys.stdout.write("tornado bind %s\n" % (options.port))
    # print("tornado bind %s" % (options.port))
    app = tornado.web.Application([
                                   (r'/dmc', DmcHandler),
                                   (r'/pid', PidHandler),
                                   (r'/customize', CustomizeHandler)
                                   ], debug=False, autoreload=False)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(tornado.options.options.port, '0.0.0.0')#int(sys.argv[1])
    # tornado.ioloop.IOLoop.initialize()
    pid = os.getpid()
    print("main pid=%d" % pid)
    http_server.start()  # num_processes=0

    th = threading.Thread(target=test)
    th.start()

    tornado.ioloop.IOLoop.instance().start()

