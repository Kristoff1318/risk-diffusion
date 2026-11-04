"""Fixed firstpath/base scalability probe; not a selected best result."""
import time,pathlib,sys,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));import panda_p0 as p
T=time.perf_counter();protocol=json.load(open(p.P/'PROTOCOL.json'));assert p.sha(p.__file__)==protocol['source_hash'];assert json.load(open(p.P/'QUALIFICATION.json'))['pass']
arm=p.Arm();z=np.load(p.P/'FROZEN_PATHS.npz');draw=p.noise(1024,protocol['seeds'][0]);row,flags,q,v=p.evaluate(arm,z['q'][0,0],z['u'][0],z['target'][0],draw)
small=np.load(p.P/'FLAGS_n256_sub10.npz')['p0_base'];assert np.array_equal(flags[:256],small)
row.update(path=0,regime='base',purpose='1024 CPU scaling probe on predetermined firstpath/base, not full bank comparison',process_seconds=time.perf_counter()-T,prefix256_identical=True)
p.save('SCALE_1024.json',row);np.savez_compressed(p.P/'SCALE_1024_FLAGS.npz',flags=flags);print(json.dumps(row),flush=True)
