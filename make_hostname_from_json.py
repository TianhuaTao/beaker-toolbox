#!/usr/bin/env python3
import json
import sys
# read json from stdin

data = json.load(sys.stdin)

data = data[0]['jobs']

hostnames = []

for job in data:
    envvars = job['execution']['spec']['envVars']
    for var_pairs in envvars:
        if var_pairs['name'] == 'BEAKER_NODE_HOSTNAME':
            hostnames.append(var_pairs['value'])
            

print('\n'.join(sorted(hostnames)))
