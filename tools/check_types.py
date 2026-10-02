"""Checks that every Get node in an .sbs reads a parameter / variable of its own
type (sbscooker fails with "Variable type mismatch" otherwise).

  python3 tools/check_types.py file.sbs
"""
import sys
import xml.etree.ElementTree as ET

GETTERS = {'get_float1': '256', 'get_float2': '512', 'get_float3': '1024', 'get_float4': '2048',
           'get_integer1': '16', 'get_integer2': '32', 'get_integer3': '64', 'get_integer4': '128',
           'get_bool': '4'}


def check(path):
    problems = []
    root = ET.parse(path).getroot()
    for g in root.iter('graph'):
        gid = g.find('identifier').get('v')
        ptypes = {p.find('identifier').get('v'): p.find('type').get('v') for p in g.findall('paraminputs/paraminput')}
        for dv in g.iter('dynamicValue'):
            sets = {}
            for pn in dv.iter('paramNode'):
                if pn.find('function').get('v') == 'set':
                    sets.setdefault(pn.find('.//constantValueString').get('v'), set()).add(pn.find('type').get('v'))
            for name, ts in sets.items():
                if len(ts) > 1:
                    problems.append('%s: variable %s set with types %s' % (gid, name, sorted(ts)))
            for pn in dv.iter('paramNode'):
                f = pn.find('function').get('v')
                if f not in GETTERS:
                    continue
                name = pn.find('.//constantValueString').get('v')
                if name.startswith('$'):
                    continue
                ts = {ptypes[name]} if name in ptypes else sets.get(name)
                if ts is None:
                    problems.append('%s: %s reads unknown variable %s' % (gid, f, name))
                elif ts != {GETTERS[f]}:
                    problems.append('%s: %s reads %s of type %s' % (gid, f, name, sorted(ts)))
    return problems


if __name__ == '__main__':
    probs = check(sys.argv[1])
    print('\n'.join(probs) or 'types ok')
    sys.exit(1 if probs else 0)
