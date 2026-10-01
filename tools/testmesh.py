import numpy as np
# cube-map layout: 6 faces in 128px tiles of a 512x512 map, 4px gutter.
FACES = {  # name: (tile col,row, axis, sign, rot90 count, mirror)
 '+X': (0,0, 0, +1, 0, False),
 '-X': (1,0, 0, -1, 1, False),
 '+Y': (2,0, 1, +1, 2, False),
 '-Y': (0,1, 1, -1, 3, False),
 '+Z': (1,1, 2, +1, 0, True),   # mirrored island
 '-Z': (2,1, 2, -1, 1, False),
}
def make(res=512, shape='sphere', tile=128, gutter=4):
    pos = np.zeros((res,res,3),np.float32); mask=np.zeros((res,res),np.float32)
    for name,(cx,cy,ax,sg,rot,mir) in FACES.items():
        s = tile*res//512; g=gutter*res//512
        x0,y0 = cx*s+g, cy*s+g; n=s-2*g
        # pixel centres -> local (a,b) in [-1,1]
        a = (np.arange(n)+0.5)/n*2-1
        A,B = np.meshgrid(a,a)  # A along x (u), B along y (v)
        for _ in range(rot): A,B = -B,A
        if mir: A=-A
        p = np.zeros((n,n,3))
        o=[i for i in range(3) if i!=ax]
        p[...,ax]=sg; p[...,o[0]]=A; p[...,o[1]]=B
        if shape=='sphere': p = p/np.linalg.norm(p,axis=-1,keepdims=True)
        elif shape=='box': p = p*np.array([1.0,0.5,0.75])
        pos[y0:y0+n,x0:x0+n]=p; mask[y0:y0+n,x0:x0+n]=1
    # dilate positions into gutter (like baked maps)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(mask==0, return_distances=False, return_indices=True)
    pos = pos[idx[0],idx[1]]
    return pos, mask
def pattern(pos):
    # sharp 3D checker + colour bands, continuous across seams
    f = np.floor(pos*3.0+100.37).astype(int)
    chk = ((f[...,0]+f[...,1]+f[...,2])%2).astype(np.float32)
    col = np.stack([chk, 0.5*chk+0.5*(pos[...,1]>0), 1-chk],-1)
    return col.astype(np.float32)
def effect(pos):
    return np.clip((pos[...,0]+1)/2,0,1).astype(np.float32)  # 0 at -X .. 1 at +X
def render(pos, mask, img, view=(0.6,0.5,0.62), res=300):
    v=np.array(view,float); v/=np.linalg.norm(v)
    up=np.array([0,1,0.]); r=np.cross(up,v); r/=np.linalg.norm(r); u=np.cross(v,r)
    m=mask>0.5; P=pos[m]; C=img[m]
    if C.ndim==1: C=np.repeat(C[:,None],3,1)
    sx=((P@r)*0.42+0.5)*res; sy=(0.5-(P@u)*0.42)*res; d=P@v
    zb=np.full(res*res,-1e9); out=np.full((res*res,3),0.15,np.float32)
    ix=np.clip(sx.astype(int),0,res-1); iy=np.clip(sy.astype(int),0,res-1)
    for dx in (0,1):
        for dy in (0,1):
            k=np.clip(iy+dy,0,res-1)*res+np.clip(ix+dx,0,res-1)
            np.maximum.at(zb,k,d)
    for dx in (0,1):
        for dy in (0,1):
            k=np.clip(iy+dy,0,res-1)*res+np.clip(ix+dx,0,res-1)
            sel=d>=zb[k]-1e-9
            out[k[sel]]=C[sel]
    return out.reshape(res,res,3)
def save(img, path):
    from PIL import Image
    a=np.clip(img,0,1)
    if a.ndim==2: a=np.repeat(a[...,None],3,2)
    Image.fromarray((a*255+0.5).astype(np.uint8)).save(path)
