import os
import uuid
import requests
import json
from glob import glob
import numpy as np
import requests
import pandas as pd

import tornado.web
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.settings import MAX_MODEL_THREAD_POOL


class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def get(self):
        self.write("App is Live!")

    def head(self):
        self.finish()

class PredictionHandler(BaseApiHandler):
    """Main Prediction Handler"""

    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, model, *args, **kwargs):
        """Initiailze the models"""
        self.model = model
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, data):
        """Blocking Call to call Predict Function"""
        # target_values = self.model.predict(X)
        # target_names = ['setosa', 'versicolor', 'virginica']
        # results = [target_names[pred] for pred in target_values]
        # return results
        print(data)

        image_filename = str(uuid.uuid4()) + ".jpg"
        image_filename = "app/static/temp-img/" + image_filename
        with open(image_filename, "wb") as fh:
            fh.write(requests.get(data["image_url"]).content)
        results = self.model.predict(image_filename)
        print(results)
        return results

    @gen.coroutine
    def predict(self, data):
        """Return prediction result"""
        print(data)
        results = yield self._blocking_predict(data)
        print("predict function", results)
        self.respond(results)
