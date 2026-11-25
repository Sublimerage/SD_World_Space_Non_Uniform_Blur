"""Checks that every Get node in an .sbs reads a parameter / variable of its own
type (sbscooker fails with "Variable type mismatch" otherwise), and that the 16 x 16
global tables are only read at their two frame texels (the builder only computes those).

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
    problems += check_frame_reads(root)
    return problems


GLOBAL_TABLES = {'WSB_Bounds', 'WSB_Coarse', 'WSB_AreaTotal', 'WSB_SortFit', 'WSB_VoxFit', 'WSNB_Radius'}
FRAME_UVS = {(4.5 / 16, 8.5 / 16), (12.5 / 16, 8.5 / 16)}


def check_frame_reads(root):
    """Every Sample node reading one of the 16 x 16 global tables must read it at a constant UV
    that is one of the two frame texels' centres."""
    problems = []
    for g in root.iter('graph'):
        gid = g.find('identifier').get('v')
        names = {o.find('GUIDependency').get('v').split('?')[1]: o.find('GUIName').get('v')
                 for o in g.iter('GUIObject')}
        small = {u for u, nm in names.items() if nm in GLOBAL_TABLES}
        nodes = list(g.find('compNodes'))
        for n in nodes:
            f = n.find('compImplementation/compFilter')
            if f is None:
                continue
            slots = set()
            for c in n.findall('connections/connection'):
                ident = c.find('identifier').get('v')
                if c.find('connRef').get('v') in small:
                    slots.add(0 if ident == 'input' else int(ident.split(':')[1]))
            if not slots:
                continue
            fn = {pn.find('uid').get('v'): pn for pn in f.iter('paramNode')}
            for pn in fn.values():
                if pn.find('function').get('v') not in ('samplecol', 'samplelum'):
                    continue
                slot = int(pn.find('.//constantValueInt2').get('v').split()[0])
                if slot not in slots:
                    continue
                src = fn[[c.find('connRef').get('v') for c in pn.findall('connections/connection')
                          if c.find('identifier').get('v') == 'pos'][0]]
                cv = src.find('.//constantValueFloat2')
                uv = tuple(float(x) for x in cv.get('v').split()) if (
                    src.find('function').get('v') == 'const_float2' and cv is not None) else None
                if uv is None or not any(abs(uv[0] - a) < 1e-6 and abs(uv[1] - b) < 1e-6 for a, b in FRAME_UVS):
                    problems.append('%s: %s reads a 16 x 16 table (slot %d) at %s, not at a frame texel'
                                    % (gid, names.get(n.find('uid').get('v'), '?'), slot, uv))
    return problems


if __name__ == '__main__':
    probs = check(sys.argv[1])
    print('\n'.join(probs) or 'types ok')
    sys.exit(1 if probs else 0)
