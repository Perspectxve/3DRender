import argparse, json, math, random
from pathlib import Path
from threading import local
import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

MATERIALS = ['grass', 'foliage', 'rock', 'stone', 'gold', 'wood', 'crystal', 'cloud']
MAT_ID = {name: index for index, name in enumerate(MATERIALS)}
MATERIAL_DENSITY = {
    'grass': 1.55,
    'foliage': 3.10,
    'rock': 1.05,
    'stone': 4.80,
    'gold': 2.80,
    'wood': 2.35,
    'crystal': 5.85,
    'cloud': 0.35,
}

# ----------------------------- utilities -----------------------------

def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize(v, eps=1e-8):
    v=np.asarray(v,dtype=np.float32); return v/(np.linalg.norm(v)+eps)

def look_at(eye, target, up=(0,1,0)):
    eye=np.asarray(eye,np.float32); target=np.asarray(target,np.float32); up=np.asarray(up,np.float32)
    z=normalize(target-eye); x=normalize(np.cross(z,up)); y=np.cross(x,z)
    R=np.stack([x,y,z],0)
    t=-R@eye
    return R,t

def sigmoid(x): return 1/(1+np.exp(-x))

def save_png(arr, path):
    arr=np.clip(arr,0,1)
    Image.fromarray((arr*255).astype(np.uint8)).save(path)

# ----------------------------- texture fields -----------------------------

def value_noise(h,w,seed=0,scale=8):
    rng=np.random.default_rng(seed)
    gh,gw=max(2,h//scale+2),max(2,w//scale+2)
    grid=rng.random((gh,gw))
    y=np.linspace(0,gh-2,h); x=np.linspace(0,gw-2,w)
    yi=np.floor(y).astype(int); xi=np.floor(x).astype(int)
    yf=y-yi; xf=x-xi
    out=np.zeros((h,w))
    for i in range(h):
        a=grid[yi[i],xi]*(1-xf)+grid[yi[i],xi+1]*xf
        b=grid[yi[i]+1,xi]*(1-xf)+grid[yi[i]+1,xi+1]*xf
        out[i]=a*(1-yf[i])+b*yf[i]
    return out

def fractal_noise(h,w,seed=0):
    layers=[]
    for octave,scale in enumerate([64,32,16,8,4]):
        layers.append(value_noise(h,w,seed+octave*17,scale)/(1.0+octave*0.65))
    noise=sum(layers)
    return (noise-noise.min())/(noise.max()-noise.min()+1e-8)

def rgba(rgb, alpha=1.0):
    if np.isscalar(alpha):
        alpha=np.full(rgb.shape[:2],alpha,np.float32)
    return np.dstack([rgb,alpha])

def make_textures(outdir, n=512):
    ensure(outdir)
    y,x=np.mgrid[0:n,0:n]/n
    tex={}
    # Painterly moss: clustered tufts, small gold flowers, and cool edge highlights.
    moss=fractal_noise(n,n,10)
    blades=np.clip(0.5+0.5*np.sin(2*np.pi*(x*55+0.22*np.sin(y*19)+moss*0.8)),0,1)
    tufts=np.clip((fractal_noise(n,n,31)-0.48)*3.2,0,1)
    flecks=(fractal_noise(n,n,48)>0.88).astype(np.float32)
    grass_rgb=np.dstack([
        0.035+0.07*moss+0.14*flecks,
        0.20+0.42*moss+0.12*blades+0.16*flecks,
        0.16+0.30*moss+0.18*tufts,
    ])
    tex['grass']=rgba(grass_rgb)

    # Cypress-like foliage: calmer vertical needle strokes with cool edge depth.
    leaf_noise=fractal_noise(n,n,57)
    fine_noise=fractal_noise(n,n,64)
    fronds=0.5+0.5*np.sin(2*np.pi*(x*46+0.12*np.sin(y*11)+leaf_noise*0.22))
    shadow=0.5+0.5*np.sin(2*np.pi*(x*3.0+0.28*np.sin(y*5.0)))
    leaf_tips=np.clip((fractal_noise(n,n,68)-0.80)*3.8,0,1)
    vertical_highlight=np.clip(1.0-y*0.55,0.35,1.0)
    foliage_rgb=np.dstack([
        0.018+0.040*fine_noise+0.030*leaf_tips,
        0.16+0.20*vertical_highlight+0.11*fronds+0.045*leaf_tips,
        0.18+0.18*leaf_noise+0.12*fronds,
    ])*(0.78+0.22*shadow[...,None])
    tex['foliage']=rgba(foliage_rgb)

    # Floating rock: directional strata and sharp indigo cracks.
    rock_noise=fractal_noise(n,n,73)
    strata=0.5+0.5*np.sin(2*np.pi*(y*13+0.18*np.sin(x*7)+rock_noise*0.30))
    cracks=np.clip((fractal_noise(n,n,91)-0.72)*5.0,0,1)
    rock_rgb=np.dstack([
        0.12+0.15*strata+0.05*rock_noise,
        0.13+0.17*strata+0.08*rock_noise,
        0.20+0.25*strata+0.14*rock_noise,
    ])*(1-0.58*cracks[...,None])
    tex['rock']=rgba(rock_rgb)

    # Celestial stone: large, high-contrast blocks that survive the splat resolve.
    rows=6.0; cols=6.0
    row=np.floor(y*rows)
    brick_x=np.mod(x*cols+0.5*np.mod(row,2),1)
    brick_y=np.mod(y*rows,1)
    edge=np.minimum(np.minimum(brick_x,1-brick_x),np.minimum(brick_y,1-brick_y))
    mortar=np.clip((0.085-edge)/0.035,0,1)
    bevel=np.clip((0.19-edge)/0.10,0,1)*(1-mortar)
    brick_tone=0.5+0.5*np.sin((np.floor(x*cols+0.5*np.mod(row,2))*17.3+row*31.7))
    stone_noise=fractal_noise(n,n,121)
    chips=np.clip((fractal_noise(n,n,138)-0.76)*4.0,0,1)
    runes=np.clip(
        (0.5+0.5*np.sin(2*np.pi*(x*5.0+0.18*np.sin(y*7.0))))*
        (0.5+0.5*np.sin(2*np.pi*(y*5.0-0.16*np.sin(x*9.0))))-0.72,
        0,1,
    )
    stone_rgb=np.dstack([
        0.36+0.20*stone_noise+0.05*brick_tone,
        0.43+0.21*stone_noise+0.05*brick_tone,
        0.54+0.24*stone_noise+0.06*brick_tone,
    ])
    stone_rgb*=1-0.78*mortar[...,None]-0.18*chips[...,None]
    stone_rgb+=bevel[...,None]*np.array([0.075,0.090,0.12],np.float32)
    stone_rgb+=np.dstack([0.005*mortar,0.035*mortar,0.055*mortar])
    stone_rgb+=runes[...,None]*np.array([0.015,0.10,0.17],np.float32)
    tex['stone']=rgba(stone_rgb)

    # Aged celestial gold: dark bronze valleys, pale filigree, and engraved lines.
    gold_noise=fractal_noise(n,n,160)
    filigree=0.5+0.5*np.sin(2*np.pi*(x*14+0.32*np.sin(y*11)))
    engraving=np.clip((0.5+0.5*np.sin(2*np.pi*(y*19+x*3)))-0.68,0,1)
    patina=np.clip((fractal_noise(n,n,177)-0.64)*2.7,0,1)
    gold_rgb=np.dstack([
        0.34+0.36*gold_noise+0.08*filigree,
        0.22+0.27*gold_noise+0.08*filigree+0.06*patina,
        0.055+0.075*gold_noise+0.10*patina,
    ])*(1-0.38*engraving[...,None])
    tex['gold']=rgba(gold_rgb)

    # Bridge wood with long grain and pale worn edges.
    grain=0.5+0.5*np.sin(2*np.pi*(x*22+0.16*np.sin(y*9)+fractal_noise(n,n,181)*0.4))
    knots=np.clip((fractal_noise(n,n,198)-0.78)*4.5,0,1)
    wood_rgb=np.dstack([
        0.20+0.27*grain,
        0.10+0.17*grain,
        0.055+0.09*grain,
    ])*(1-0.38*knots[...,None])
    tex['wood']=rgba(wood_rgb)

    # Luminous crystal RGBA: mostly unified cyan with just enough facet variation
    # to keep the silhouettes readable without looking noisy or rainbow-banded.
    facet=0.5+0.5*np.sin(2*np.pi*(x*7+y*15+0.28*np.sin(x*17)))
    core=np.clip(1.0-np.abs(x-0.5)*2.0,0,1)
    crystal_rgb=np.dstack([
        0.045+0.020*facet+0.014*core,
        0.620+0.055*facet+0.045*core,
        0.925+0.035*facet+0.020*core,
    ])
    crystal_alpha=np.clip(0.18+0.060*facet+0.055*core,0.18,0.34)
    tex['crystal']=rgba(crystal_rgb,crystal_alpha)

    cloud_noise=fractal_noise(n,n,220)
    cloud_rgb=np.dstack([
        0.66+0.22*cloud_noise,
        0.75+0.20*cloud_noise,
        0.92+0.08*cloud_noise,
    ])
    tex['cloud']=rgba(cloud_rgb,0.025+0.055*cloud_noise)

    for name,img in tex.items():
        tex[name]=np.clip(img,0,1); save_png(tex[name], Path(outdir)/f"{name}.png")
    return tex

def sample_tex(tex,u,v):
    h,w,_=tex.shape
    ui=(np.abs(u)%1.0*(w-1)).astype(int); vi=(np.abs(v)%1.0*(h-1)).astype(int)
    return tex[vi,ui]

# ----------------------------- mesh generation -----------------------------

class Mesh:
    def __init__(self): self.v=[]; self.c=[]; self.f=[]; self.name=[]
    def add_vertices(self, verts, colors):
        base=len(self.v); self.v.extend(np.asarray(verts,np.float32)); self.c.extend(np.asarray(colors,np.float32)); return base
    def add_faces(self, faces, mat='default'):
        self.f.extend(faces); self.name.extend([mat]*len(faces))
    def merge(self, other):
        base=len(self.v); self.v.extend(other.v); self.c.extend(other.c); self.f.extend([(a+base,b+base,c+base) for a,b,c in other.f]); self.name.extend(other.name)
    def arrays(self): return np.array(self.v,np.float32), np.array(self.c,np.float32), np.array(self.f,np.int32)

def add_box(mesh, center, size, color, mat='box'):
    cx,cy,cz=center; sx,sy,sz=np.array(size)/2
    verts=np.array([[cx+dx*sx,cy+dy*sy,cz+dz*sz] for dx in[-1,1] for dy in[-1,1] for dz in[-1,1]],np.float32)
    faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    b=mesh.add_vertices(verts, np.tile(color,(8,1))); mesh.add_faces([(a+b,c+b,d+b) for a,c,d in faces], mat)

def add_oriented_box(mesh, center, size, yaw, color, mat='box'):
    cx,cy,cz=center; sx,sy,sz=np.array(size)/2
    local=np.array([[dx*sx,dy*sy,dz*sz] for dx in[-1,1] for dy in[-1,1] for dz in[-1,1]],np.float32)
    ca,sa=np.cos(yaw),np.sin(yaw)
    rot=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]],np.float32)
    verts=local@rot.T+np.array([cx,cy,cz],np.float32)
    faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    b=mesh.add_vertices(verts,np.tile(color,(8,1))); mesh.add_faces([(a+b,c+b,d+b) for a,c,d in faces],mat)

def add_cylinder(mesh, center, radius, height, color, seg=18, mat='cyl'):
    cx,cy,cz=center; verts=[]; cols=[]
    for y in [-height/2,height/2]:
        for i in range(seg):
            a=2*np.pi*i/seg; verts.append([cx+radius*np.cos(a),cy+y,cz+radius*np.sin(a)]); cols.append(color)
    verts.append([cx,cy-height/2,cz]); cols.append(color); bot=2*seg
    verts.append([cx,cy+height/2,cz]); cols.append(color); top=2*seg+1
    b=mesh.add_vertices(verts,cols); faces=[]
    for i in range(seg):
        j=(i+1)%seg; faces += [(b+i,b+j,b+seg+j),(b+i,b+seg+j,b+seg+i),(b+bot,b+j,b+i),(b+top,b+seg+i,b+seg+j)]
    mesh.add_faces(faces, mat)

def add_cylinder_between(mesh, start, end, radius, color, seg=12, mat='cyl'):
    start=np.asarray(start,np.float32); end=np.asarray(end,np.float32)
    axis=end-start; length=float(np.linalg.norm(axis))
    if length<1e-5: return
    up=axis/length
    helper=np.array([0,1,0],np.float32) if abs(up[1])<0.92 else np.array([1,0,0],np.float32)
    side=normalize(np.cross(up,helper)); forward=normalize(np.cross(side,up))
    verts=[]; cols=[]
    for p in [start,end]:
        for i in range(seg):
            a=2*np.pi*i/seg
            verts.append(p+radius*(side*np.cos(a)+forward*np.sin(a))); cols.append(color)
    b=mesh.add_vertices(verts,cols); faces=[]
    for i in range(seg):
        j=(i+1)%seg
        faces += [(b+i,b+j,b+seg+j),(b+i,b+seg+j,b+seg+i)]
    mesh.add_faces(faces,mat)

def add_torus(mesh, center, major, minor, color, major_seg=48, minor_seg=8, mat='gold'):
    cx,cy,cz=center; verts=[]; cols=[]
    for i in range(major_seg):
        a=2*np.pi*i/major_seg
        for j in range(minor_seg):
            b=2*np.pi*j/minor_seg
            r=major+minor*np.cos(b)
            verts.append([cx+r*np.cos(a),cy+minor*np.sin(b),cz+r*np.sin(a)])
            cols.append(color)
    base=mesh.add_vertices(verts,cols); faces=[]
    for i in range(major_seg):
        ii=(i+1)%major_seg
        for j in range(minor_seg):
            jj=(j+1)%minor_seg
            a=base+i*minor_seg+j; b=base+ii*minor_seg+j
            c=base+ii*minor_seg+jj; d=base+i*minor_seg+jj
            faces += [(a,b,c),(a,c,d)]
    mesh.add_faces(faces,mat)

def add_ellipsoid(mesh, center, radii, color, rings=8, seg=16, mat='cloud'):
    cx,cy,cz=center; rx,ry,rz=radii; verts=[]; cols=[]
    for r in range(1,rings):
        phi=np.pi*r/rings
        for i in range(seg):
            a=2*np.pi*i/seg
            verts.append([cx+rx*np.sin(phi)*np.cos(a),cy+ry*np.cos(phi),cz+rz*np.sin(phi)*np.sin(a)])
            cols.append(color)
    verts += [[cx,cy+ry,cz],[cx,cy-ry,cz]]; cols += [color,color]
    base=mesh.add_vertices(verts,cols); top=base+(rings-1)*seg; bottom=top+1; faces=[]
    for r in range(rings-2):
        for i in range(seg):
            j=(i+1)%seg; a=base+r*seg+i; b=base+r*seg+j
            c=base+(r+1)*seg+j; d=base+(r+1)*seg+i
            faces += [(a,b,c),(a,c,d)]
    for i in range(seg):
        j=(i+1)%seg
        faces += [(top,base+j,base+i),(bottom,base+(rings-2)*seg+i,base+(rings-2)*seg+j)]
    mesh.add_faces(faces,mat)

def add_cone(mesh, center, radius, height, color, seg=16, mat='cone'):
    cx,cy,cz=center; verts=[]; cols=[]
    for i in range(seg):
        a=2*np.pi*i/seg; verts.append([cx+radius*np.cos(a),cy-height/2,cz+radius*np.sin(a)]); cols.append(color)
    verts.append([cx,cy+height/2,cz]); cols.append(color); apex=seg
    verts.append([cx,cy-height/2,cz]); cols.append(color); basec=seg+1
    b=mesh.add_vertices(verts,cols); faces=[]
    for i in range(seg):
        j=(i+1)%seg; faces += [(b+i,b+j,b+apex),(b+basec,b+j,b+i)]
    mesh.add_faces(faces,mat)

def add_crystal(mesh, center, radius, height, color, seg=6):
    cx,cy,cz=center; verts=[]; cols=[]
    mid=cy; top=cy+height*0.55; bot=cy-height*0.45
    for i in range(seg):
        a=2*np.pi*i/seg; verts.append([cx+radius*np.cos(a),mid,cz+radius*np.sin(a)]); cols.append(color*(0.75+0.25*(i%2)))
    verts += [[cx,top,cz],[cx,bot,cz]]; cols += [np.clip(color*1.4,0,1),color*0.55]
    b=mesh.add_vertices(verts,cols); faces=[]
    for i in range(seg):
        j=(i+1)%seg; faces += [(b+i,b+j,b+seg),(b+j,b+i,b+seg+1)]
    mesh.add_faces(faces,'crystal')

def add_stylized_tree(mesh, center, scale=1.0, lean=0.0):
    """Build a sparse celestial cypress with a visible trunk and layered foliage."""
    cx,cy,cz=center
    wood=np.array([0.30,0.16,0.08],np.float32)
    foliage=np.array([0.045,0.30,0.34],np.float32)
    trunk_top=np.array([cx+lean*scale,cy+1.42*scale,cz],np.float32)
    add_cylinder_between(
        mesh,
        np.array([cx,cy,cz],np.float32),
        trunk_top,
        0.085*scale,
        wood,
        16,
        'wood',
    )
    for height,rx,ry,tint in [
        (0.78,0.48,0.62,0.78),
        (1.18,0.40,0.70,0.92),
        (1.58,0.28,0.56,1.08),
        (1.92,0.15,0.34,1.18),
    ]:
        add_ellipsoid(
            mesh,
            (cx+lean*scale*height/1.92,cy+height*scale,cz),
            (rx*scale,ry*scale,rx*scale),
            np.clip(foliage*tint,0,1),
            10,
            20,
            'foliage',
        )

def add_bridge(mesh, start, end, width=0.52, sag=0.22, planks=22):
    start=np.asarray(start,np.float32); end=np.asarray(end,np.float32)
    direction=end-start; yaw=math.atan2(direction[0],direction[2])
    wood=np.array([0.42,0.22,0.08]); gold=np.array([0.78,0.48,0.10])
    positions=[]
    for i in range(planks):
        t=i/max(planks-1,1)
        p=start*(1-t)+end*t
        p[1]-=sag*4*t*(1-t)
        positions.append(p.copy())
        add_oriented_box(mesh,p,(width,0.07,0.16),yaw,wood,'wood')
    left=np.array([math.cos(yaw),0,-math.sin(yaw)],np.float32)*width*0.48
    for side in [-1,1]:
        rail=[p+left*side+np.array([0,0.30,0],np.float32) for p in positions]
        for a,b in zip(rail[:-1],rail[1:]):
            add_cylinder_between(mesh,a,b,0.018,gold,8,'gold')
        for p in positions[::3]:
            add_cylinder_between(mesh,p+left*side,p+left*side+np.array([0,0.30,0],np.float32),0.014,gold,8,'gold')

def add_arch(mesh, center, radius, leg_height, color, tube=0.07, segments=24, mat='stone'):
    cx,cy,cz=center
    points=[
        np.array([cx+radius*np.cos(a),cy+radius*np.sin(a),cz],np.float32)
        for a in np.linspace(0,np.pi,segments+1)
    ]
    for a,b in zip(points[:-1],points[1:]):
        add_cylinder_between(mesh,a,b,tube,color,12,mat)
    for p in [points[0],points[-1]]:
        add_cylinder_between(mesh,p,p-np.array([0,leg_height,0],np.float32),tube,color,12,mat)

def add_brick_ring(mesh, center, radius, height, color, segments=24):
    """Create a stepped altar course whose brick joints remain visible as geometry."""
    cx,cy,cz=center
    grout=np.asarray(color,np.float32)*np.array([0.24,0.28,0.38],np.float32)
    add_cylinder(mesh,(cx,cy,cz),radius,height,grout,max(24,segments*2),'rock')
    block_radius=radius+0.035
    arc_width=2*np.pi*block_radius/segments*0.82
    for i in range(segments):
        a=2*np.pi*i/segments
        variation=0.90+0.10*np.sin(i*2.37+radius*5.0)
        add_oriented_box(
            mesh,
            (cx+block_radius*np.cos(a),cy,cz+block_radius*np.sin(a)),
            (arc_width,height*0.78,0.14),
            np.pi/2-a,
            np.clip(np.asarray(color)*variation,0,1),
            'stone',
        )
    add_cylinder(mesh,(cx,cy+height*0.52,cz),radius*0.96,height*0.08,np.asarray(color)*1.03,max(24,segments*2),'stone')

def add_grand_altar(mesh, center, scale=1.0):
    cx,cy,cz=center; stone=np.array([0.48,0.59,0.72]); gold=np.array([0.72,0.50,0.16])
    # Broad ceremonial dais and a luminous central crystal.
    for k,(radius,height,segments) in enumerate([(1.55,0.18,30),(1.25,0.17,26),(0.96,0.16,22),(0.68,0.15,18)]):
        add_brick_ring(mesh,(cx,cy+0.09+k*0.16,cz),radius*scale,height*scale,stone*(0.90+0.04*k),segments)
    # Descending approach steps terminate on the island before the front bridge.
    for k,(offset,height) in enumerate([(1.24,0.125),(1.40,0.090),(1.56,0.055),(1.72,0.020)]):
        add_box(mesh,(cx,cy+height*scale,cz-offset*scale),(1.35*scale,0.07*scale,0.24*scale),stone*(0.86-0.025*k),'stone')
    add_torus(mesh,(cx,cy+0.68*scale,cz),0.86*scale,0.045*scale,gold,56,8,'gold')
    add_arch(mesh,(cx,cy+1.02*scale,cz+0.88*scale),1.28*scale,0.88*scale,stone,0.080*scale,30,'stone')
    add_arch(mesh,(cx,cy+1.02*scale,cz+0.84*scale),1.08*scale,0.78*scale,gold,0.032*scale,28,'gold')
    add_crystal(mesh,(cx,cy+1.35*scale,cz),0.34*scale,2.55*scale,np.array([0.08,0.72,1.0]),8)
    # Four guardian columns keep the altar ceremonial without crowding it.
    for i in range(4):
        a=np.pi/4+2*np.pi*i/4; x=cx+1.20*scale*np.cos(a); z=cz+1.20*scale*np.sin(a)
        add_cylinder(mesh,(x,cy+0.64*scale,z),0.065*scale,0.90*scale,stone,18,'stone')
        add_crystal(mesh,(x,cy+1.16*scale,z),0.075*scale,0.46*scale,np.array([0.10,0.68,1.0]),6)

def island(mesh, center, rx, rz, top_y, depth, tex, seed=0, n=22):
    rng=np.random.default_rng(seed); cx,cy,cz=center
    verts=[]; cols=[]; idx={};
    grass=tex['grass']; rock=tex['rock']
    for i in range(n):
        for j in range(n):
            x=-1+2*i/(n-1); z=-1+2*j/(n-1); r=(x*x+z*z)**0.5
            if r<=1.05:
                h=0.20*np.sin(6*x+seed)+0.13*np.cos(5*z+seed*0.7)+0.08*rng.normal()
                fade=max(0,1-r**2)
                p=[cx+x*rx, top_y+h*fade, cz+z*rz]
                col=sample_tex(grass,np.array([(x+1)/2*3]),np.array([(z+1)/2*3]))[0,:3]
                idx[(i,j)]=len(verts); verts.append(p); cols.append(col)
    faces=[]
    for i in range(n-1):
        for j in range(n-1):
            if all((a,b) in idx for a,b in [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]):
                faces += [(idx[(i,j)],idx[(i+1,j)],idx[(i+1,j+1)]),(idx[(i,j)],idx[(i+1,j+1)],idx[(i,j+1)])]
    base=mesh.add_vertices(verts,cols); mesh.add_faces([(a+base,b+base,c+base) for a,b,c in faces],'grass')
    # underside rings tapering to point
    rings=8; seg=48; verts=[]; cols=[]
    for k in range(rings):
        t=k/(rings-1); rad=(1-t)**1.8; y=top_y-0.08-depth*t
        for q in range(seg):
            a=2*np.pi*q/seg; rr=0.9+0.08*np.sin(q*3+seed)
            verts.append([cx+rx*rad*rr*np.cos(a), y+0.06*np.sin(q*5+k), cz+rz*rad*rr*np.sin(a)])
            cols.append(sample_tex(rock,np.array([q/seg*4]),np.array([t*4]))[0,:3])
    b=mesh.add_vertices(verts,cols); faces=[]
    for k in range(rings-1):
        for q in range(seg):
            qq=(q+1)%seg
            faces += [(b+k*seg+q,b+k*seg+qq,b+(k+1)*seg+qq),(b+k*seg+q,b+(k+1)*seg+qq,b+(k+1)*seg+q)]
    mesh.add_faces(faces,'rock')

def build_scene(tex):
    mesh=Mesh()
    islands=[
        ((0,0,0),3.45,2.70,0.0,2.65),
        ((-5.55,-0.45,0.95),1.75,1.30,-0.28,1.55),
        ((5.55,-0.42,0.80),1.85,1.35,-0.24,1.60),
        ((0.15,-0.52,-4.25),1.65,1.25,-0.36,1.45),
        ((0.75,-0.68,4.65),1.55,1.15,-0.50,1.30),
    ]
    for s,it in enumerate(islands):
        island(mesh,*it,tex,seed=s+2,n=46 if s==0 else 30)

    # The open altar owns the center; sparse trees replace the former roofed temple.
    add_grand_altar(mesh,(0,0.04,-0.62),1.12)
    add_bridge(mesh,(-3.15,-0.13,0.55),(-4.12,-0.22,0.76),0.56,0.18,18)
    add_bridge(mesh,(3.15,-0.13,0.48),(4.12,-0.20,0.70),0.56,0.18,18)
    add_bridge(mesh,(0.10,-0.18,-2.78),(0.14,-0.31,-3.18),0.64,0.10,10)
    add_bridge(mesh,(1.85,-0.18,2.22),(1.30,-0.38,3.58),0.58,0.20,22)

    # A small grove frames the shrine without closing off the camera path.
    for center,scale,lean in [
        ((-1.80,0.08,1.62),0.90,-0.08),
        ((-2.72,0.04,0.48),0.72,0.05),
        ((2.42,0.05,1.42),0.82,0.08),
        ((-5.55,-0.24,1.05),0.74,-0.05),
        ((5.55,-0.20,0.86),0.78,0.05),
        ((0.82,-0.42,4.72),0.68,-0.04),
    ]:
        add_stylized_tree(mesh,center,scale,lean)

    # Crystals around islands and along the ceremonial approach.
    rng=np.random.default_rng(4)
    for k in range(8):
        ang=rng.uniform(0,2*np.pi); rad=rng.uniform(0.85,2.85); x=rad*np.cos(ang); z=rad*0.76*np.sin(ang); y=0.12+rng.uniform(0,0.28)
        add_crystal(mesh,(x,y,z),rng.uniform(0.055,0.13),rng.uniform(0.42,0.95),np.array([0.10,0.70,1.0]),6)
    for basec,rx,rz,ty,dep in islands[1:]:
        cx,cy,cz=basec
        for k in range(2):
            ang=rng.uniform(0,2*np.pi); r=rng.uniform(0.2,0.9)
            add_crystal(mesh,(cx+rx*r*np.cos(ang),ty+0.14,cz+rz*r*np.sin(ang)),0.07,0.48,np.array([0.12,0.66,1.0]),6)
    for z in np.linspace(-2.20,-1.20,3):
        add_crystal(mesh,(-0.72,0.18,z),0.07,0.48,np.array([0.08,0.72,1.0]),6)
        add_crystal(mesh,(0.72,0.18,z),0.07,0.48,np.array([0.08,0.72,1.0]),6)

    # Translucent cloud banks frame the scene without becoming opaque boxes.
    for k in range(16):
        a=rng.uniform(0,2*np.pi); rad=rng.uniform(5.8,9.0)
        x=rad*np.cos(a); z=rad*np.sin(a)+1.5; y=rng.uniform(-2.45,-1.25); s=rng.uniform(0.30,0.68)
        add_ellipsoid(mesh,(x,y,z),(s*1.9,s*0.42,s),np.array([0.72,0.82,1.0]),7,14,'cloud')
    return mesh

# ----------------------------- export and sampling -----------------------------

def export_obj(mesh,outdir):
    ensure(outdir); V,C,F=mesh.arrays(); obj=Path(outdir)/'sky_sanctuary_teacher_scene.obj'; ply=Path(outdir)/'sky_sanctuary_teacher_scene.ply'
    mtl=Path(outdir)/'sky_sanctuary_teacher_scene.mtl'
    material_defs={
        'grass':('0.08 0.42 0.28','1.0','0.00 0.00 0.00'),
        'foliage':('0.04 0.28 0.34','1.0','0.00 0.00 0.00'),
        'rock':('0.20 0.22 0.34','1.0','0.00 0.00 0.00'),
        'stone':('0.52 0.58 0.68','1.0','0.00 0.00 0.00'),
        'gold':('0.92 0.60 0.12','1.0','0.08 0.03 0.00'),
        'wood':('0.35 0.18 0.07','1.0','0.00 0.00 0.00'),
        'crystal':('0.06 0.62 1.00','0.20','0.00 0.25 0.65'),
        'cloud':('0.72 0.82 1.00','0.12','0.00 0.00 0.00'),
    }
    with open(mtl,'w') as f:
        for name,(kd,alpha,ke) in material_defs.items():
            f.write(f'newmtl {name}\nKa 0.03 0.04 0.06\nKd {kd}\nKs 0.18 0.24 0.32\nKe {ke}\nNs 48\nd {alpha}\nillum 2\n\n')
    with open(obj,'w') as f:
        f.write('# teacher scene, vertex colors stored as extra RGB fields\n')
        f.write(f'mtllib {mtl.name}\n')
        for v,c in zip(V,C): f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}\n")
        active=None
        for (a,b,c),name in zip(F+1,mesh.name):
            if name!=active:
                active=name; f.write(f'usemtl {name}\n')
            f.write(f"f {a} {b} {c}\n")
    with open(ply,'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v,c in zip(V,C): f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")
        for a,b,c in F: f.write(f"3 {a} {b} {c}\n")
    return obj,ply

def material_rgba(points, normals, material_ids, tex):
    result=np.zeros((len(points),4),np.float32)
    for name,mat_id in MAT_ID.items():
        mask=material_ids==mat_id
        if not np.any(mask): continue
        p=points[mask]; n=normals[mask]
        if name=='grass':
            u=p[:,0]*0.55; v=p[:,2]*0.55
        elif name=='foliage':
            u=p[:,0]*1.15+p[:,2]*0.34
            v=p[:,1]*0.82
        elif name=='rock':
            u=np.arctan2(p[:,2],p[:,0])/(2*np.pi)*4.0; v=p[:,1]*1.65
        elif name=='stone':
            vertical=np.abs(n[:,1])<0.68
            u=np.where(vertical,(p[:,0]+p[:,2])*0.28,p[:,0]*0.30)
            v=np.where(vertical,p[:,1]*0.48,p[:,2]*0.30)
        elif name=='gold':
            vertical=np.abs(n[:,1])<0.68
            u=np.where(vertical,(p[:,0]-p[:,2])*0.62,p[:,0]*0.72)
            v=np.where(vertical,p[:,1]*1.10,p[:,2]*0.72)
        elif name=='wood':
            u=(p[:,0]+p[:,2])*1.70; v=(p[:,0]-p[:,2])*0.42
        elif name=='crystal':
            u=np.arctan2(n[:,2],n[:,0])/(2*np.pi)*2.0; v=p[:,1]*1.25
        else:
            u=p[:,0]*0.30; v=p[:,2]*0.30
        result[mask]=sample_tex(tex[name],u,v)
    return result

def allocate_face_samples(area, names, count):
    density=np.asarray([MATERIAL_DENSITY.get(str(name),1.0) for name in names],np.float64)
    weights=area.astype(np.float64)*density
    expected=weights/np.maximum(weights.sum(),1e-8)*count
    counts=np.floor(expected).astype(np.int32)
    # Architectural faces need multiple representatives so texture does not
    # collapse into one flat-colored splat on small blocks and railings.
    names=np.asarray(names)
    minimum=np.zeros(len(names),np.int32)
    minimum[names!='cloud']=1
    minimum[names=='foliage']=4
    minimum[names=='stone']=8
    minimum[names=='gold']=4
    minimum[names=='wood']=3
    minimum[names=='crystal']=28
    counts=np.maximum(counts,minimum)
    delta=count-int(counts.sum())
    if delta>0:
        remainder=expected-np.floor(expected)
        order=np.argsort(remainder)[::-1]
        counts[order[:delta]]+=1
    elif delta<0:
        removable=np.flatnonzero(counts>minimum)
        order=removable[np.argsort(counts[removable])[::-1]]
        for index in order:
            take=min(counts[index]-minimum[index],-delta); counts[index]-=take; delta+=take
            if delta==0: break
    if delta<0:
        raise RuntimeError(f'splat budget {count} is too small for required face coverage')
    return counts

def sample_splats(mesh, tex, count=120000):
    """Stratified, material-aware sampling into surface-aligned Gaussian splats."""
    V,C,F=mesh.arrays(); tri=V[F]; col=C[F]; names=np.asarray(mesh.name)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(cross,axis=1)/2
    normals=cross/np.maximum(np.linalg.norm(cross,axis=1,keepdims=True),1e-8)
    edges=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,1],tri[:,0]-tri[:,2]],axis=1)
    lengths=np.linalg.norm(edges,axis=2)
    tangent=edges[np.arange(len(edges)),np.argmax(lengths,axis=1)]
    tangent=tangent/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
    counts=allocate_face_samples(area,names,count)

    points=[]; base_colors=[]; out_normals=[]; tangents=[]; scales=[]; material_ids=[]
    golden=0.6180339887498949
    for face_id,n in enumerate(counts):
        if n<=0: continue
        q=np.arange(n,dtype=np.float32)
        r1=np.sqrt(np.mod((q+0.5)/max(n,1)+face_id*0.17320508,1.0))[:,None]
        r2=np.mod((q+0.5)*golden+face_id*0.41421356,1.0)[:,None]
        a=1-r1; b=r1*(1-r2); c=r1*r2
        points.append(a*tri[face_id,0]+b*tri[face_id,1]+c*tri[face_id,2])
        base_colors.append(a*col[face_id,0]+b*col[face_id,1]+c*col[face_id,2])
        out_normals.append(np.repeat(normals[face_id][None,:],n,axis=0))
        tangents.append(np.repeat(tangent[face_id][None,:],n,axis=0))
        spacing=math.sqrt(max(float(area[face_id])/max(n,1),1e-8))
        mat=str(names[face_id]); factor=0.50
        if mat=='crystal': factor=0.30
        elif mat=='cloud': factor=1.00
        elif mat=='foliage': factor=0.42
        elif mat=='stone': factor=0.43
        elif mat in ('gold','wood'): factor=0.40
        scales.append(np.repeat(np.array([[spacing*factor,spacing*factor,spacing*0.12]],np.float32),n,axis=0))
        material_ids.append(np.full(n,MAT_ID.get(mat,0),np.int16))

    P=np.concatenate(points).astype(np.float32)
    vertex_rgb=np.concatenate(base_colors).astype(np.float32)
    N=np.concatenate(out_normals).astype(np.float32)
    T=np.concatenate(tangents).astype(np.float32)
    scale=np.concatenate(scales).astype(np.float32)
    material_id=np.concatenate(material_ids).astype(np.int16)
    texture=material_rgba(P,N,material_id,tex)
    rgb=np.clip(texture[:,:3]*0.88+vertex_rgb*0.12,0,1).astype(np.float32)
    opacity=texture[:,3].astype(np.float32)
    opacity[material_id==MAT_ID['grass']]=np.clip(opacity[material_id==MAT_ID['grass']],0.92,0.99)
    opacity[material_id==MAT_ID['rock']]=np.clip(opacity[material_id==MAT_ID['rock']],0.94,0.995)
    opacity[material_id==MAT_ID['stone']]=np.clip(opacity[material_id==MAT_ID['stone']],0.94,0.995)
    opacity[material_id==MAT_ID['gold']]=np.clip(opacity[material_id==MAT_ID['gold']],0.92,0.99)
    opacity[material_id==MAT_ID['wood']]=np.clip(opacity[material_id==MAT_ID['wood']],0.94,0.995)
    opacity[material_id==MAT_ID['crystal']]=np.clip(opacity[material_id==MAT_ID['crystal']],0.12,0.36)
    opacity[material_id==MAT_ID['cloud']]=np.clip(opacity[material_id==MAT_ID['cloud']],0.02,0.08)
    emissive=np.zeros(len(P),np.float32)
    emissive[material_id==MAT_ID['crystal']]=0.95
    emissive[material_id==MAT_ID['gold']]=0.10
    return {
        'xyz':P,'rgb':rgb,'scale':scale,'opacity':opacity,'normal':N,'tangent':T,
        'material_id':material_id,'emissive':emissive,
    }

# ----------------------------- rendering -----------------------------

def sky_background(W,H):
    yy=np.linspace(0,1,H,dtype=np.float32)[:,None]
    xx=np.linspace(0,1,W,dtype=np.float32)[None,:]
    top=np.array([0.012,0.025,0.090],np.float32)
    horizon=np.array([0.10,0.24,0.42],np.float32)
    bg=top[None,None,:]*(1-yy[:,:,None])+horizon[None,None,:]*yy[:,:,None]
    bg=np.repeat(bg,W,axis=1)
    aurora=np.exp(-((yy-0.62-0.035*np.sin(xx*8.0))**2)/0.010)
    bg+=aurora[:,:,None]*np.array([0.02,0.11,0.16],np.float32)
    sun=np.exp(-((xx-0.78)**2+(yy-0.18)**2)/0.0024)
    bg+=sun[:,:,None]*np.array([0.42,0.31,0.10],np.float32)
    rng=np.random.default_rng(77)
    for _ in range(max(40,W*H//9000)):
        x=int(rng.integers(0,W)); y=int(rng.integers(0,max(1,int(H*0.62))))
        strength=float(rng.uniform(0.15,0.65)); bg[y,x]+=np.array([0.35,0.55,0.85])*strength
    return np.clip(bg,0,1)

def render_splats_np(scene, eye, target, W=640,H=360, fov=55, bg=None, glow=True):
    P=scene['xyz']; RGB=scene['rgb']; scales=scene['scale']; opacity=scene['opacity']
    normals=scene['normal']; tangents=scene['tangent']; emissive=scene['emissive']; material=scene['material_id']
    if bg is None: bg=sky_background(W,H)
    R,t=look_at(eye,target); Pc=(R@P.T).T+t
    z=Pc[:,2]; mask=z>0.12
    Pc=Pc[mask]; rgb=RGB[mask]; sc=scales[mask]; op=opacity[mask]; z=z[mask]
    nrm=normals[mask]; tan=tangents[mask]; emit=emissive[mask]; mat=material[mask]
    bit=np.cross(nrm,tan); bit/=np.maximum(np.linalg.norm(bit,axis=1,keepdims=True),1e-8)
    axis_u=tan*sc[:,0:1]; axis_v=bit*sc[:,1:2]
    au=axis_u@R.T; av=axis_v@R.T
    f=0.5*W/math.tan(math.radians(fov)/2)
    u=f*Pc[:,0]/z+W/2; v=-f*Pc[:,1]/z+H/2
    du1=f*(au[:,0]*z-Pc[:,0]*au[:,2])/np.maximum(z*z,1e-8)
    dv1=-f*(au[:,1]*z-Pc[:,1]*au[:,2])/np.maximum(z*z,1e-8)
    du2=f*(av[:,0]*z-Pc[:,0]*av[:,2])/np.maximum(z*z,1e-8)
    dv2=-f*(av[:,1]*z-Pc[:,1]*av[:,2])/np.maximum(z*z,1e-8)
    c00=du1*du1+du2*du2+0.20
    c01=du1*dv1+du2*dv2
    c11=dv1*dv1+dv2*dv2+0.20
    trace=c00+c11; disc=np.sqrt(np.maximum((c00-c11)**2+4*c01*c01,0))
    radius=np.clip(3.0*np.sqrt(np.maximum((trace+disc)*0.5,0.20)),1.0,30.0)
    m=(u>-32)&(u<W+32)&(v>-32)&(v<H+32)
    u=u[m]; v=v[m]; z=z[m]; rgb=rgb[m]; op=op[m]; emit=emit[m]; mat=mat[m]; sc=sc[m]
    c00=c00[m]; c01=c01[m]; c11=c11[m]; radius=radius[m]
    img=bg.copy().astype(np.float32); emission=np.zeros_like(img)
    zbuffer=np.full((H,W),np.inf,np.float32)
    transparent=(mat==MAT_ID['crystal'])|(mat==MAT_ID['cloud'])

    # Solid splats are accumulated only near the first visible surface. This is
    # a surface-splat resolve: it removes point-cell boundaries while keeping
    # foreground and background geometry separated.
    solid_rgb=np.zeros_like(img)
    solid_emission=np.zeros_like(img)
    solid_weight=np.zeros((H,W),np.float32)
    for idx in np.flatnonzero(~transparent)[np.argsort(z[~transparent])]:
        px=float(u[idx]); py=float(v[idx]); size=int(math.ceil(radius[idx]))
        x0=max(0,int(math.floor(px-size))); x1=min(W,int(math.ceil(px+size+1)))
        y0=max(0,int(math.floor(py-size))); y1=min(H,int(math.ceil(py+size+1)))
        if x0>=x1 or y0>=y1: continue
        det=c00[idx]*c11[idx]-c01[idx]*c01[idx]
        if det<1e-7: continue
        inv00=c11[idx]/det; inv01=-c01[idx]/det; inv11=c00[idx]/det
        xs=np.arange(x0,x1,dtype=np.float32)-px; ys=np.arange(y0,y1,dtype=np.float32)-py
        q=inv00*xs[None,:]**2+2*inv01*ys[:,None]*xs[None,:]+inv11*ys[:,None]**2
        inside=q<6.25
        if not np.any(inside): continue
        depth_patch=zbuffer[y0:y1,x0:x1]
        fresh=inside&np.isinf(depth_patch)
        depth_patch[fresh]=z[idx]
        depth_tolerance=float(np.clip(sc[idx,0]*1.8,0.035,0.18))
        visible=inside&(z[idx]<=depth_patch+depth_tolerance)
        if not np.any(visible): continue
        weight=np.where(visible,np.exp(-0.5*q)*op[idx],0.0).astype(np.float32)
        solid_rgb[y0:y1,x0:x1]+=rgb[idx]*weight[...,None]
        solid_weight[y0:y1,x0:x1]+=weight
        if emit[idx]>0.01:
            solid_emission[y0:y1,x0:x1]+=rgb[idx]*(weight[...,None]*emit[idx])
    covered=solid_weight>1e-5
    img[covered]=solid_rgb[covered]/solid_weight[covered][:,None]
    emission[covered]=solid_emission[covered]/solid_weight[covered][:,None]

    # Find the nearest crystal shell before alpha compositing. Without this
    # surface-depth resolve, every front and back crystal splat accumulates into
    # an opaque-looking cyan volume and the faceted silhouette disappears.
    crystal_depth=np.full((H,W),np.inf,np.float32)
    for idx in np.flatnonzero(mat==MAT_ID['crystal'])[np.argsort(z[mat==MAT_ID['crystal']])]:
        px=float(u[idx]); py=float(v[idx]); size=int(math.ceil(radius[idx]))
        x0=max(0,int(math.floor(px-size))); x1=min(W,int(math.ceil(px+size+1)))
        y0=max(0,int(math.floor(py-size))); y1=min(H,int(math.ceil(py+size+1)))
        if x0>=x1 or y0>=y1: continue
        det=c00[idx]*c11[idx]-c01[idx]*c01[idx]
        if det<1e-7: continue
        inv00=c11[idx]/det; inv01=-c01[idx]/det; inv11=c00[idx]/det
        xs=np.arange(x0,x1,dtype=np.float32)-px; ys=np.arange(y0,y1,dtype=np.float32)-py
        q=inv00*xs[None,:]**2+2*inv01*ys[:,None]*xs[None,:]+inv11*ys[:,None]**2
        visible=(q<6.25)&(z[idx]<=zbuffer[y0:y1,x0:x1]+0.06)
        if np.any(visible):
            patch=crystal_depth[y0:y1,x0:x1]
            patch[visible]=np.minimum(patch[visible],z[idx])

    # Crystals and clouds keep their alpha channels and composite back-to-front.
    for idx in np.flatnonzero(transparent)[np.argsort(z[transparent])[::-1]]:
        px=float(u[idx]); py=float(v[idx]); size=int(math.ceil(radius[idx]))
        x0=max(0,int(math.floor(px-size))); x1=min(W,int(math.ceil(px+size+1)))
        y0=max(0,int(math.floor(py-size))); y1=min(H,int(math.ceil(py+size+1)))
        if x0>=x1 or y0>=y1: continue
        det=c00[idx]*c11[idx]-c01[idx]*c01[idx]
        if det<1e-7: continue
        inv00=c11[idx]/det; inv01=-c01[idx]/det; inv11=c00[idx]/det
        xs=np.arange(x0,x1,dtype=np.float32)-px; ys=np.arange(y0,y1,dtype=np.float32)-py
        q=inv00*xs[None,:]**2+2*inv01*ys[:,None]*xs[None,:]+inv11*ys[:,None]**2
        visible=(q<9.0)&(z[idx]<=zbuffer[y0:y1,x0:x1]+0.06)
        if mat[idx]==MAT_ID['crystal']:
            shell_tolerance=float(np.clip(sc[idx,0]*2.2,0.035,0.14))
            visible&=z[idx]<=crystal_depth[y0:y1,x0:x1]+shell_tolerance
        if not np.any(visible): continue
        a=np.where(visible,op[idx]*np.exp(-0.5*q),0.0).astype(np.float32)
        patch=img[y0:y1,x0:x1]
        img[y0:y1,x0:x1]=patch*(1-a[...,None])+rgb[idx]*a[...,None]
        if emit[idx]>0.01:
            emission[y0:y1,x0:x1]+=rgb[idx]*(a[...,None]*emit[idx])
    if glow and np.any(emission>0):
        e=np.uint8(np.clip(emission,0,1)*255)
        small=np.asarray(Image.fromarray(e).filter(ImageFilter.GaussianBlur(radius=4)),dtype=np.float32)/255
        wide=np.asarray(Image.fromarray(e).filter(ImageFilter.GaussianBlur(radius=14)),dtype=np.float32)/255
        img=np.clip(img+small*0.75+wide*0.38,0,1)
    return img

# ----------------------------- lightweight Gaussian training -----------------------------

def build_teacher_targets(scene):
    P=scene['xyz']; base=scene['rgb']; N=scene['normal']; material=scene['material_id']
    key=normalize(np.array([-0.45,0.82,-0.34],np.float32))
    fill=normalize(np.array([0.60,0.30,0.72],np.float32))
    diffuse=np.clip(N@key,0,1)[:,None]
    cool=np.clip(N@fill,0,1)[:,None]
    height=np.clip((P[:,1:2]+2.0)/5.0,0,1)
    light=0.34+0.55*diffuse+0.13*cool+0.10*height
    target=np.clip(base*light,0,1)

    crystal=material==MAT_ID['crystal']; gold=material==MAT_ID['gold']; cloud=material==MAT_ID['cloud']
    target[gold]=np.clip(target[gold]+np.array([0.06,0.030,0.00],np.float32),0,1)
    crystal_tint=np.array([0.055,0.690,1.000],np.float32)
    target[crystal]=np.clip(base[crystal]*0.44+crystal_tint*0.56+np.array([0.00,0.035,0.075],np.float32),0,1)
    target[cloud]=np.clip(base[cloud]*(0.82+0.18*height[cloud]),0,1)

    # Fake crystal point-light: distance falloff + normal-facing term.
    main_light_pos = np.array([0.0, 1.35, -0.62], np.float32)

    to_light = main_light_pos[None, :] - P
    dist = np.linalg.norm(to_light, axis=1, keepdims=True)
    light_dir = to_light / np.maximum(dist, 1e-6)

    # Surfaces facing the crystal get more light.
    facing = np.clip(np.sum(N * light_dir, axis=1, keepdims=True), 0.0, 1.0)

    # Distance falloff. Smaller denominator = shorter/stronger light.
    falloff = 1.0 / (1.0 + 0.55 * dist * dist)

    # Emphasize stone/gold/grass/wood; do not over-light crystal/cloud.
    receiver = (
        (material == MAT_ID['stone']) |
        (material == MAT_ID['gold']) |
        (material == MAT_ID['grass']) |
        (material == MAT_ID['wood']) |
        (material == MAT_ID['rock'])
    )[:, None].astype(np.float32)

    crystal_light = receiver * falloff * (0.35 + 0.65 * facing)

    target = np.clip(
        target + crystal_light * np.array([0.035, 0.18, 0.34], np.float32),
        0,
        1
    )

    # Extra soft light pool under the main crystal.
    light_pool = np.exp(-((P[:,0:1] - 0.0)**2 + (P[:,2:3] + 0.62)**2) / 1.15)
    top_faces = np.clip(N[:,1:2], 0.0, 1.0)
    target = np.clip(
        target + receiver * light_pool * top_faces * np.array([0.02, 0.12, 0.24], np.float32),
        0,
        1
    )
    altar_light=np.exp(-np.linalg.norm(P-np.array([0.0,1.05,-0.62],np.float32),axis=1)[:,None]*0.82)
    target=np.clip(target+altar_light*np.array([0.018,0.105,0.20],np.float32),0,1)

    target_op=scene['opacity'].copy()
    target_emissive=scene['emissive'].copy()
    crystal_height=np.clip((P[crystal,1]-0.1)/2.8,0,1)
    target_op[crystal]=np.clip(target_op[crystal]*(1.02+0.95*crystal_height),0.18,0.50)
    target_emissive[crystal]=np.clip(0.78+0.22*crystal_height,0.72,1.00)
    return target.astype(np.float32),target_op.astype(np.float32),target_emissive.astype(np.float32)

def train_gaussian_representation(mesh,tex,outdir,mode='quick'):
    """Fit a residual neural material model over a dense surface-splat teacher."""
    ensure(outdir); view_dir=Path(outdir)/'teacher_views'; ensure(view_dir)
    for old_view in view_dir.glob('view_*.png'):
        old_view.unlink()
    views=8
    count=220000 if mode=='quick' else 1250000
    scene=sample_splats(mesh,tex,count=count)
    target_rgb,target_op,target_emissive=build_teacher_targets(scene)

    # Save multi-view inspection renders from the textured, surface-aligned teacher.
    teacher_scene={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in scene.items()}
    teacher_scene['rgb']=target_rgb; teacher_scene['opacity']=target_op; teacher_scene['emissive']=target_emissive
    teacher_render_scene=teacher_scene
    if mode=='full' and len(scene['xyz'])>420000:
        selected=np.linspace(0,len(scene['xyz'])-1,420000,dtype=np.int64)
        teacher_render_scene={
            key:(value[selected] if isinstance(value,np.ndarray) and len(value)==len(scene['xyz']) else value)
            for key,value in teacher_scene.items()
        }
        teacher_render_scene['scale']=teacher_render_scene['scale'].copy()
        teacher_render_scene['scale'][:,:2]*=1.02
    for i in tqdm(range(views),desc='rendering dense teacher views'):
        th=2*np.pi*i/views; rad=8.3+0.55*np.sin(i*1.7)
        eye=np.array([rad*np.sin(th),2.0+0.42*np.sin(2*th),rad*np.cos(th)+0.7],np.float32)
        target=np.array([0,0.65,0.15],np.float32)
        img=render_splats_np(teacher_render_scene,eye,target,360 if mode=='quick' else 560,204 if mode=='quick' else 315,glow=True)
        save_png(img,view_dir/f'view_{i:03d}.png')

    device='cpu'
    xyz=torch.tensor(scene['xyz']/8.0,dtype=torch.float32,device=device)
    normal=torch.tensor(scene['normal'],dtype=torch.float32,device=device)
    base=torch.tensor(scene['rgb'],dtype=torch.float32,device=device)
    mat=torch.nn.functional.one_hot(torch.tensor(scene['material_id'],dtype=torch.int64),len(MATERIALS)).float()
    target=torch.tensor(target_rgb,dtype=torch.float32,device=device)
    target_op_t=torch.tensor(target_op[:,None],dtype=torch.float32,device=device)
    target_em_t=torch.tensor(target_emissive[:,None],dtype=torch.float32,device=device)

    class MatNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dim=3*6*2+3+3+len(MATERIALS)
            self.net=torch.nn.Sequential(
                torch.nn.Linear(dim,96),torch.nn.SiLU(),
                torch.nn.Linear(96,96),torch.nn.SiLU(),
                torch.nn.Linear(96,5))
        def forward(self,x,n,c,m):
            feats=[]
            for f in [1,2,4,8,16,32]:
                feats += [torch.sin(f*x),torch.cos(f*x)]
            out=self.net(torch.cat(feats+[n,c,m],dim=-1))
            rgb=torch.clamp(c+torch.tanh(out[:,:3])*0.34,0,1)
            return rgb,torch.sigmoid(out[:,3:4]),torch.sigmoid(out[:,4:5])

    net=MatNet().to(device)
    optm=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-5)
    steps=80 if mode=='quick' else 260; bs=4096
    for _ in tqdm(range(steps),desc='fitting neural material attributes for splats'):
        idx=torch.randint(0,len(xyz),(bs,),device=device)
        pr,po,pe=net(xyz[idx],normal[idx],base[idx],mat[idx])
        loss=((pr-target[idx])**2).mean()+0.30*((po-target_op_t[idx])**2).mean()+0.22*((pe-target_em_t[idx])**2).mean()
        optm.zero_grad(); loss.backward(); optm.step()
    with torch.no_grad():
        learned_rgb=[]; learned_op=[]; learned_em=[]
        for a in range(0,len(xyz),16384):
            pr,po,pe=net(xyz[a:a+16384],normal[a:a+16384],base[a:a+16384],mat[a:a+16384])
            learned_rgb.append(pr.cpu()); learned_op.append(po.cpu()); learned_em.append(pe.cpu())
        scene['rgb']=torch.cat(learned_rgb).numpy().astype(np.float32)
        scene['opacity']=torch.cat(learned_op).numpy().squeeze().astype(np.float32)
        scene['emissive']=torch.cat(learned_em).numpy().squeeze().astype(np.float32)

    crystal=scene['material_id']==MAT_ID['crystal']; cloud=scene['material_id']==MAT_ID['cloud']; solid=~(crystal|cloud)
    scene['opacity'][solid]=np.clip(0.62*scene['opacity'][solid]+0.38*target_op[solid],0.86,0.995)
    crystal_tint=np.array([0.055,0.690,1.000],np.float32)
    scene['rgb'][crystal]=np.clip(scene['rgb'][crystal]*0.55+crystal_tint*0.45,0,1)
    scene['opacity'][crystal]=np.clip(0.42*scene['opacity'][crystal]+0.58*target_op[crystal],0.18,0.50)
    scene['opacity'][cloud]=np.clip(0.30*scene['opacity'][cloud]+0.70*target_op[cloud],0.015,0.08)
    scene['emissive'][crystal]=np.clip(0.20*scene['emissive'][crystal]+0.80*target_emissive[crystal],0.68,1.00)
    scene['emissive'][~crystal]*=0.18
    np.savez_compressed(Path(outdir)/'learned_gaussian_splats.npz',**scene)
    torch.save(net.state_dict(),Path(outdir)/'neural_material_mlp.pt')
    stats={'teacher_views':views,'material_training_steps':steps,'num_splats':len(scene['xyz']),'final_loss':float(loss.detach())}
    return scene,stats

# ----------------------------- video and output -----------------------------

def camera_path(n):
    eye_keys=np.asarray([
        [0.0,0.85,-9.0],
        [0.0,0.95,-6.2],
        [-0.35,1.10,-3.8],
        [-2.8,1.85,-1.2],
        [-3.8,2.55,2.1],
        [0.2,3.35,4.8],
        [3.4,2.45,2.1],
    ],np.float32)
    target_keys=np.asarray([
        [0.0,0.45,-0.8],
        [0.0,0.65,-0.6],
        [0.0,1.00,-0.5],
        [0.0,1.20,-0.45],
        [0.0,1.10,0.20],
        [0.0,1.00,0.75],
        [0.0,1.05,0.35],
    ],np.float32)
    def sample(keys,t):
        progress=t*(len(keys)-1); i=min(int(progress),len(keys)-2)
        u=progress-i; s=u*u*(3-2*u)
        return keys[i]*(1-s)+keys[i+1]*s
    return [(sample(eye_keys,i/max(n-1,1)),sample(target_keys,i/max(n-1,1))) for i in range(n)]

def video_settings(mode):
    frames=36 if mode=='quick' else 100
    W,H=(640,360) if mode=='quick' else (960,540)
    fps=18 if mode=='quick' else 20
    return frames,W,H,fps

def video_render_scene(scene,mode):
    render_scene=scene
    if mode=='full' and len(scene['xyz'])>900000:
        selected=np.linspace(0,len(scene['xyz'])-1,900000,dtype=np.int64)
        render_scene={key:(value[selected] if isinstance(value,np.ndarray) and len(value)==len(scene['xyz']) else value) for key,value in scene.items()}
    return render_scene

def render_video_frames(scene,outdir,mode,start=0,end=None):
    ensure(outdir); frames,W,H,_=video_settings(mode)
    end=frames if end is None else min(end,frames)
    if start<0 or start>=end:
        raise ValueError(f'invalid frame range [{start}, {end}) for {frames} frames')
    frame_dir=Path(outdir)/'frames'; ensure(frame_dir)
    render_scene=video_render_scene(scene,mode)
    render_scene = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in render_scene.items()
    }

    stone = render_scene['material_id'] == MAT_ID['stone']
    rock = render_scene['material_id'] == MAT_ID['rock']
    grass = render_scene['material_id'] == MAT_ID['grass']
    wood = render_scene['material_id'] == MAT_ID['wood']
    gold = render_scene['material_id'] == MAT_ID['gold']
    foliage = render_scene['material_id'] == MAT_ID['foliage']

    render_scene['scale'][stone, :2] *= 1.10
    render_scene['scale'][rock, :2] *= 1.10
    render_scene['scale'][grass, :2] *= 1.08
    render_scene['scale'][wood, :2] *= 1.06
    render_scene['scale'][gold, :2] *= 1.00
    render_scene['scale'][foliage, :2] *= 1.06
    path=camera_path(frames)
    for k in tqdm(range(start,end),desc=f'rendering Gaussian-splat frames {start}:{end}'):
        eye,target=path[k]
        img=render_splats_np(render_scene,eye,target,W,H,fov=52,glow=True)
        save_png(img,frame_dir/f'frame_{k:04d}.png')

def encode_video(outdir,mode):
    frames,_,_,fps=video_settings(mode)
    frame_dir=Path(outdir)/'frames'
    missing=[k for k in range(frames) if not (frame_dir/f'frame_{k:04d}.png').exists()]
    if missing:
        raise RuntimeError(f'cannot encode video; missing {len(missing)} frame(s), beginning with {missing[:5]}')
    writer=imageio.get_writer(Path(outdir)/'final_gaussian_splat_flythrough.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=None)
    for k in tqdm(range(frames),desc='encoding final Gaussian-splat video'):
        frame=np.asarray(Image.open(frame_dir/f'frame_{k:04d}.png').convert('RGB'))
        writer.append_data(frame)
        if k==0:
            Image.fromarray(frame).save(Path(outdir)/'preview.png')
    writer.close()

def rotation_quaternions(normals,tangents):
    n=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    t=tangents-n*np.sum(tangents*n,axis=1,keepdims=True)
    t=t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8)
    b=np.cross(n,t)
    matrices=np.stack([t,b,n],axis=2)
    q=np.zeros((len(matrices),4),np.float32)
    for i,m in enumerate(matrices):
        trace=float(np.trace(m))
        if trace>0:
            s=math.sqrt(trace+1.0)*2; q[i]=[0.25*s,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s]
        else:
            j=int(np.argmax(np.diag(m)))
            if j==0:
                s=math.sqrt(max(1.0+m[0,0]-m[1,1]-m[2,2],1e-8))*2
                q[i]=[(m[2,1]-m[1,2])/s,0.25*s,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s]
            elif j==1:
                s=math.sqrt(max(1.0+m[1,1]-m[0,0]-m[2,2],1e-8))*2
                q[i]=[(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,0.25*s,(m[1,2]+m[2,1])/s]
            else:
                s=math.sqrt(max(1.0+m[2,2]-m[0,0]-m[1,1],1e-8))*2
                q[i]=[(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,0.25*s]
    return q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)

def export_splats_ply(scene,path):
    P=scene['xyz']; RGB=np.clip(scene['rgb'],0,1); N=scene['normal']; scale=scene['scale']
    opacity=np.clip(scene['opacity'],1e-4,1-1e-4); q=rotation_quaternions(N,scene['tangent'])
    dtype=np.dtype([
        ('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),
        ('f_dc_0','<f4'),('f_dc_1','<f4'),('f_dc_2','<f4'),('opacity','<f4'),
        ('scale_0','<f4'),('scale_1','<f4'),('scale_2','<f4'),
        ('rot_0','<f4'),('rot_1','<f4'),('rot_2','<f4'),('rot_3','<f4'),
        ('red','u1'),('green','u1'),('blue','u1'),('alpha','u1'),
        ('emissive','<f4'),('material_id','<i2'),
    ])
    data=np.empty(len(P),dtype=dtype)
    for name,column in [('x',0),('y',1),('z',2)]: data[name]=P[:,column]
    for name,column in [('nx',0),('ny',1),('nz',2)]: data[name]=N[:,column]
    sh=(RGB-0.5)/0.28209479177387814
    for name,column in [('f_dc_0',0),('f_dc_1',1),('f_dc_2',2)]: data[name]=sh[:,column]
    data['opacity']=np.log(opacity/(1-opacity))
    logs=np.log(np.maximum(scale,1e-5))
    for name,column in [('scale_0',0),('scale_1',1),('scale_2',2)]: data[name]=logs[:,column]
    for name,column in [('rot_0',0),('rot_1',1),('rot_2',2),('rot_3',3)]: data[name]=q[:,column]
    rgb8=np.uint8(np.round(RGB*255))
    data['red']=rgb8[:,0]; data['green']=rgb8[:,1]; data['blue']=rgb8[:,2]
    data['alpha']=np.uint8(np.round(opacity*255)); data['emissive']=scene['emissive']; data['material_id']=scene['material_id']
    header=(
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(P)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property float nx\nproperty float ny\nproperty float nz\n'
        'property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\n'
        'property float scale_0\nproperty float scale_1\nproperty float scale_2\n'
        'property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n'
        'property float emissive\nproperty short material_id\nend_header\n'
    ).encode('ascii')
    with open(path,'wb') as f:
        f.write(header); data.tofile(f)

def load_scene(path):
    with np.load(path) as data:
        return {key:data[key] for key in data.files}

def write_summary(out,mode,mesh,scene,stats):
    ply_path=out/'gaussian_scene'/'learned_gaussian_splats.ply'
    obj_path=out/'teacher_scene'/'sky_sanctuary_teacher_scene.obj'
    content_size=(ply_path.stat().st_size+obj_path.stat().st_size)/(1024*1024)
    summary={
        'project':'Sky Sanctuary GS: stylized floating-island sanctuary represented as learned Gaussian surfaces',
        'mode':mode,
        'teacher_mesh_vertices':len(mesh.v),'teacher_mesh_faces':len(mesh.f),'gaussian_splats':len(scene['xyz']),
        'materials':MATERIALS,
        'crystal_alpha_range':[float(scene['opacity'][scene['material_id']==MAT_ID['crystal']].min()),float(scene['opacity'][scene['material_id']==MAT_ID['crystal']].max())],
        'ai_ml_components':['procedural surface-attribute supervision','learned anisotropic Gaussian-surface representation','PyTorch residual neural material model','learned splat color, opacity, and emissive attributes','synthetic multi-view inspection renders'],
        'rendering':'final video rendered by the custom NumPy anisotropic Gaussian-splat renderer; no third-party 3D renderer',
        '3d_content_mib':content_size,
        'training':stats
    }
    with open(out/'scene_summary.json','w') as f: json.dump(summary,f,indent=2)
    if content_size>100:
        raise RuntimeError(f'3D content exceeds 100 MiB: {content_size:.2f} MiB')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['quick','full'],default='quick')
    ap.add_argument('--out',default='outputs')
    ap.add_argument('--stage',choices=['all','build','render','encode'],default='all')
    ap.add_argument('--frame-start',type=int,default=0)
    ap.add_argument('--frame-end',type=int)
    args=ap.parse_args(); out=Path(args.out); ensure(out)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    scene=None
    if args.stage in ('all','build'):
        tex=make_textures(out/'textures',1024 if args.mode=='full' else 256)
        mesh=build_scene(tex)
        export_obj(mesh,out/'teacher_scene')
        scene,stats=train_gaussian_representation(mesh,tex,out/'gaussian_scene',args.mode)
        export_splats_ply(scene,out/'gaussian_scene'/'learned_gaussian_splats.ply')
        write_summary(out,args.mode,mesh,scene,stats)
    if args.stage in ('all','render'):
        if scene is None:
            scene=load_scene(out/'gaussian_scene'/'learned_gaussian_splats.npz')
        render_video_frames(scene,out,args.mode,args.frame_start,args.frame_end)
    if args.stage in ('all','encode'):
        encode_video(out,args.mode)
    print('\nDone. Key outputs:')
    if args.stage in ('all','encode'):
        print(' ',out/'final_gaussian_splat_flythrough.mp4')
    print(' ',out/'gaussian_scene'/'learned_gaussian_splats.npz')
    print(' ',out/'gaussian_scene'/'learned_gaussian_splats.ply')
    print(' ',out/'teacher_scene'/'sky_sanctuary_teacher_scene.obj')

if __name__=='__main__': main()
