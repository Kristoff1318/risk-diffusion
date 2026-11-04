import time, pathlib,sys,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));import panda_p0 as p
arm=p.Arm();z=np.load(p.P/'FROZEN_PATHS.npz');d=tuple(v[:64] for v in p.noise(1024,2026103000));rows=[];flags=[]
for chunk in [256,512,1024]:
 row,f,_,_=p.evaluate(arm,z['q'][0,0],z['u'][0],z['target'][0],d,10,chunk);rows.append(row);flags.append(f);print(json.dumps({'chunk':chunk,'seconds':row['runtime_seconds'],'RSS_KiB':row['peak_RSS_KiB']}),flush=True)
assert all(np.array_equal(flags[0],f) for f in flags)
p.save('CHUNK_PROFILE.json',{'rows':rows,'identical_event_flags':True,'scope':'fixed firstpath/base, no outcome selection; memory peak is process cumulative'})
