import xmltodict
import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
import sys

with open('annotations.xml') as fd:
    doc = xmltodict.parse(fd.read())

doc["annotations"].keys()


with open(sys.argv[1]) as fd:
    doc = xmltodict.parse(fd.read())

data_list = []
for anno in doc["annotations"]["image"]:
    data = {}
    data['width'] = int(anno['@width'])
    data['height'] = int(anno['@height'])
    data['name'] = anno['@name']
    data['stitches'] = []
    data['incisions'] = []
    if "polyline" in anno:
        if not isinstance(anno["polyline"], list):
            pline = anno["polyline"]
            pts = [[float(x) for x in pt.split(",")] for pt in anno["polyline"]["@points"].split(";")]
            if pline['@label'] == 'Incision':
                data['incisions'].append(pts)
            if pline['@label'] == 'Stitch':
                data['stitches'].append(pts)

        else:
            for pline in anno["polyline"]:
                pts = [[float(x) for x in pt.split(",")] for pt in pline["@points"].split(";")]
                if pline['@label'] == 'Incision':
                    data['incisions'].append(pts)
                if pline['@label'] == 'Stitch':
                    data['stitches'].append(pts)

    data_list.append(data)

with open(sys.argv[2],'w') as fw:
   json.dump(data_list, fw, indent=4)