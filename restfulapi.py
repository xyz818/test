#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import requests
from urlparse import urljoin

BASE_URL = 'http://192.168.0.177:18080/RabbitMqServer/deviceinfo/LXG001'



def test_get_user_list():
    rsp = requests.get(BASE_URL)
    return rsp.content




if __name__=='__main__':
    str =  test_get_user_list()
    data = json.loads(str)
    print data['pi_seq']