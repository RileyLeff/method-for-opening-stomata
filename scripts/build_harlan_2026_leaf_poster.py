"""Print poster: generated botanical background, real data, vector labels."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LightSource
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import qrcode
import math
from fontTools.ttLib import TTFont as FontToolsFont
from fontTools.varLib.instancer import instantiateVariableFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from photo26.control_plots import load_control
import build_harlan_2026 as base
from poster_warp import to_out

ROOT = base.ROOT
TMP = base.TMP / 'leaf'
TMP.mkdir(exist_ok=True)
ASSETS = base.OUT / 'assets'
W, H = 3456, 2592
# A second Morella leaf held at ambient CO2 (no CO2 steps), started at 17:56.
CONTROL_PATH = ROOT / 'data/2026-06-22-1756_logdata_morcer_indoor_overnight'
# Gulf Coast reference (Nguyen 2013): mean, SE.
REFERENCE = {'Vcmax25': (84, 6), 'Jmax25': (116, 10)}

def areal_fonts():
    """Freeze variable font instances so PDF and matplotlib use identical outlines."""
    folder = TMP / 'fonts'
    folder.mkdir(exist_ok=True)
    for style, weight, slant in [('Regular',400,0), ('Bold',700,0),
                                  ('Italic',400,-12), ('BoldItalic',700,-12)]:
        path = folder / f'Areal-{style}.ttf'
        if not path.exists():
            source = FontToolsFont('/Users/rileyleff/Library/Fonts/ABCArealSuperfamilyVariable.ttf')
            font = instantiateVariableFont(source, {'MONO':0,'wght':weight,'slnt':slant,'DRKM':0}, inplace=True)
            family = 'Areal Poster'
            subfamily = style.replace('BoldItalic', 'Bold Italic')
            for record in font['name'].names:
                replacement = {1:family, 2:subfamily, 4:f'{family} {subfamily}',
                               6:f'ArealPoster-{style}',16:family,17:subfamily}.get(record.nameID)
                if replacement is not None:
                    record.string = replacement.encode(record.getEncoding())
            font.save(path)
        font_manager.fontManager.addfont(path)
        pdfmetrics.registerFont(TTFont(f'Areal-{style}', str(path)))
    mpl.rcParams.update({'font.family':'Areal Poster', 'text.color':'#142C23',
                         'axes.labelcolor':'#142C23', 'xtick.color':'#243B30',
                         'ytick.color':'#243B30', 'axes.edgecolor':'#243B30',
                         'axes.linewidth':1.6, 'xtick.major.width':1.6, 'ytick.major.width':1.6,
                         'xtick.major.size':8, 'ytick.major.size':8,
                         'mathtext.fontset':'custom',
                         'mathtext.rm':'Areal Poster', 'mathtext.it':'Areal Poster:italic',
                         'mathtext.bf':'Areal Poster:bold', 'mathtext.sf':'Areal Poster'})

def knockout(path, enclosed=False):
    """Copy of a logo with its white surround made transparent. Only white
    connected to the border goes, so white lettering inside a mark survives;
    enclosed=True also clears white pockets (for pure line art)."""
    from scipy import ndimage as ndi
    im = Image.open(path).convert('RGBA')
    a = np.asarray(im).copy()
    white = a[..., :3].min(axis=2) > 235
    lab, _ = ndi.label(white)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = white if enclosed else np.isin(lab, edge[edge > 0])
    # Feather the cut so JPEG fringes don't leave a halo.
    soft = ndi.gaussian_filter(bg.astype(float), 1.0)
    a[..., 3] = (a[..., 3] * (1 - soft)).astype(np.uint8)
    out = TMP / f'knockout-{path.stem}.png'
    # Trim to the mark itself so every logo is sized by its visible extent.
    im = Image.fromarray(a)
    im.crop(im.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()).save(out)
    return out

def save(fig, name):
    # dpi only affects rasterized artists (the 3D surface); 600 ppi at print size.
    fig.savefig(TMP / f'{name}.pdf', transparent=True, dpi=600)
    plt.close(fig)

def axes_in(fig, x, y, w, h):
    """Axes placed in inches from the figure's bottom-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes((x/fw, y/fh, w/fw, h/fh))

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#243B30', alpha=.12, lw=1.2)
    ax.tick_params(labelsize=26, pad=8)

def co2_strip(ax, minute, ca, stages):
    """Chamber CO2 trace above a time-series panel, instead of a second y-axis.
    Stages are labelled with their set-points."""
    ax.fill_between(minute, 0, ca, color=base.ORANGE, alpha=.16, lw=0)
    ax.plot(minute, ca, color=base.ORANGE, lw=3.5)
    ax.set(xlim=(0,120), ylim=(0,560), xticks=[], yticks=[])
    ax.spines[['top','right','left','bottom']].set_visible(False)
    for stage in stages:
        ax.text((stage['start'] + min(stage['end'], 120)) / 2, stage['co2'] + 35, f"{stage['co2']} ppm",
                ha='center', va='bottom', fontsize=21, color=base.INK)

def method_plot():
    data = json.loads((base.ASSETS / 'low-co2-opening.json').read_text())
    df = pd.DataFrame(data['records'])
    control = load_control(CONTROL_PATH)
    control = control.loc[control.minute.le(120)]
    fig = plt.figure(figsize=(45, 7.2))
    main_h, strip_y, strip_h = 4.15, 5.75, .95
    gsw_label = r'$g_{sw}\;(\mathrm{mol\,m^{-2}\,s^{-1}})$'+'\nStomatal openness'
    time_label = r'$t\;(\mathrm{min})$, Elapsed time'
    for x, w, col, color, label, ylim, yticks in [
            (3.2, 10.3, 'gsw', base.BLUE, gsw_label, (0,.24), [0,.1,.2]),
            (27.3, 15.8, 'a', base.CYAN, r'$A\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nPhotosynthetic rate',
             (0,17), [0,5,10,15])]:
        strip = axes_in(fig, x, strip_y, w, strip_h)
        co2_strip(strip, df.minute, df.ca, data['stages'])
        strip.text(-.012, .3, 'Chamber\nCO₂', transform=strip.transAxes, ha='right', va='center',
                   fontsize=21, color=base.INK, linespacing=1.1)
        ax = axes_in(fig, x, 1.45, w, main_h)
        for boundary in [30,60,90]:
            for a in (ax, strip):
                a.axvline(boundary, color=base.SECONDARY, lw=1.6, ls=(0,(5,5)), alpha=.55)
        ax.plot(df.minute, df[col], color=color, lw=5, solid_capstyle='round')
        ax.set(xlim=(0,120), ylim=ylim, xticks=[0,30,60,90,120], yticks=yticks)
        ax.set_xlabel(time_label, fontsize=26, labelpad=10)
        ax.set_ylabel(label, fontsize=26, labelpad=14)
        clean(ax)
        if col == 'a':
            initial = df.loc[df.minute.ge(25) & df.minute.lt(30), 'a'].mean()
            peak = df.loc[df.minute.ge(90) & df.minute.le(95), 'a'].max()
            for value, name, xpos in [(initial,'Initial 420 ppm',.02),(peak,'Return peak',.505)]:
                ax.axhline(value,color=base.SECONDARY,lw=2,ls=(0,(3,4)),alpha=.85)
                ax.text(xpos,value+.4,name,transform=ax.get_yaxis_transform(),
                        color=base.INK,fontsize=21,va='bottom')
    # Unmanipulated second leaf: same light and VPD, CO2 never stepped down.
    x, w = 15.6, 4.3
    strip = axes_in(fig, x, strip_y, w, strip_h)
    co2_strip(strip, control.minute, control.Ca, [{'start':0, 'end':120, 'co2':420}])
    strip.text(.5, 1.14, 'Unmanipulated leaf', transform=strip.transAxes, ha='center',
               va='bottom', fontsize=22, fontweight='bold', color=base.INK)
    ax = axes_in(fig, x, 1.45, w, main_h)
    ax.plot(control.minute, control.gsw, color=base.SECONDARY, lw=5, solid_capstyle='round')
    # Same y-scale as the treatment leaf so the two can be compared directly.
    ax.set(xlim=(0,120), ylim=(0,.24), xticks=[0,60,120], yticks=[0,.1,.2])
    ax.set_xlabel(time_label, fontsize=26, labelpad=10)
    clean(ax)
    save(fig, 'method')

def capacity_plot():
    import xarray as xr
    from scipy.stats import gaussian_kde
    tree = xr.open_datatree(ROOT/'outputs/phrag_brownsville_tier_c_posterior.nc')
    draws, _ = base.build_draws_and_summary(tree)
    tree.close()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.3))
    fig.subplots_adjust(left=.05, right=.97, top=.97, bottom=.25, wspace=.22)
    groups = [('High', base.BLUE, 'Low salinity (n = 3)'), ('Low', base.ORANGE, 'High salinity (n = 5)')]
    for ax, p, limit in zip(axes, ['Vcmax25','Jmax25'], [130, 220]):
        ref, se = REFERENCE[p]
        xx = np.linspace(0, limit, 800)
        top = 0
        for zone, color, label in groups:
            yy = gaussian_kde(draws[p][zone])(xx)
            ax.fill_between(xx, 0, yy, color=color, alpha=.2, lw=0)
            ax.plot(xx, yy, color=color, lw=4.5, label=label)
            top = max(top, yy.max())
        ax.set_ylim(0, top * 1.3)
        ax.axvspan(ref - se, ref + se, color=base.INK, alpha=.08, lw=0)
        ax.axvline(ref, color=base.INK, lw=2.4, ls=(0,(5,4)))
        ax.text(ref + se + limit * .015, top * 1.27, 'Gulf Coast\n(Nguyen 2013)', ha='left', va='top',
                fontsize=21, color=base.INK, linespacing=1.1)
        ax.set_xlim(0, limit)
        ax.set_yticks([])
        if p == 'Vcmax25':
            name, units = r'$V_{cmax,25}$', r' ($\mu$mol m$^{-2}$ s$^{-1}$)'
            title = 'Carbon fixation capacity'
        else:
            name, units = r'$J_{max,25}$', r' ($\mu$mol e$^{-}$ m$^{-2}$ s$^{-1}$)'
            title = 'Electron transport capacity'
        ax.set_xlabel(name + units + '\n' + title, fontsize=25, labelpad=11)
        ax.tick_params(labelsize=26, pad=8)
        ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_ylabel('Posterior density', fontsize=23, labelpad=10)
    # Direct labels on the outer flank of each curve in the Vcmax panel.
    ax = axes[0]
    xx = np.linspace(0, 130, 800)
    top = ax.get_ylim()[1] / 1.3
    for (zone, color, label), side, text_xy in zip(groups, ['left','right'], [(0, .86), (50, .5)]):
        yy = gaussian_kde(draws['Vcmax25'][zone])(xx)
        peak = int(yy.argmax())
        flank = np.arange(peak) if side == 'left' else np.arange(peak, len(xx))
        i = flank[np.argmin(np.abs(yy[flank] - yy[peak] * (.88 if side == 'left' else .5)))]
        name, n = label.split(' (')
        ax.annotate(f'{name}\n({n}', xy=(xx[i], yy[i]), xytext=(text_xy[0], text_xy[1] * top * 1.3),
                    ha='left', va='bottom' if side == 'left' else 'center', fontsize=22,
                    fontweight='bold', color=base.INK, linespacing=1.1,
                    arrowprops={'arrowstyle':'-', 'color':color, 'lw':3, 'shrinkA':6, 'shrinkB':2})
    save(fig, 'capacity')

def surface_plot():
    data = json.loads((base.ASSETS/'phrag-p10-surface.json').read_text())
    ci, q = np.meshgrid(data['ci'], data['q'])
    z = np.array(data['assimilation'])
    fig = plt.figure(figsize=(16, 8.4))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    shade = LightSource(azdeg=225, altdeg=40).shade(
        z, cmap=mpl.colormaps['viridis'], vmin=0, vmax=26, blend_mode='soft',
        vert_exag=.06, dx=float(np.ptp(ci))/z.shape[1], dy=float(np.ptp(q))/z.shape[0])
    ax.plot_surface(ci, q, z, facecolors=shade, rcount=z.shape[0], ccount=z.shape[1],
                    linewidth=0, antialiased=False, shade=False, rasterized=True)
    # Light iso-lines on the surface so A can be read off without the colour scale.
    for level in [5, 10, 15, 20, 25]:
        ax.contour(ci, q, z, levels=[level], colors='white', linewidths=1.6, alpha=.75)
    ax.set(xlim=(0,max(data['ci'])),ylim=(0,2000),zlim=(0,27),xticks=[0,600,1200],yticks=[0,1000,2000],zticks=[0,10,20])
    ax.set_xlabel(r'$C_i\;(\mathrm{\mu mol\,mol^{-1}})$'+'\nInternal CO₂',fontsize=22,labelpad=22)
    ax.set_ylabel(r'$Q\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nIncident light',fontsize=22,labelpad=25)
    ax.set_zlabel(r'$A\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nPhotosynthetic rate',fontsize=22,labelpad=18)
    ax.tick_params(labelsize=22,pad=3)
    ax.view_init(elev=24,azim=-135)
    ax.set_box_aspect((1.35,1.15,.7),zoom=.95)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_pane_color((.95,.98,.95,0))
        axis._axinfo['grid']['color']=(.3,.4,.35,.18)
        axis._axinfo['grid']['linewidth']=1.2
    fig.subplots_adjust(left=0,right=1,bottom=.06,top=1.02)
    save(fig,'surface')

def npq_plot():
    leaves = pd.read_csv(ROOT/'outputs/phrag_brownsville_npq_overall_leaf_means.csv')
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(left=.2, right=.95, bottom=.27, top=.8)
    for x, zone, color in [(0,'High',base.BLUE),(1,'Low',base.ORANGE)]:
        vals = leaves.loc[leaves.zone.eq(zone),'NPQ'].to_numpy()
        ax.hlines(vals.mean(), x-.24, x+.24, color=color, lw=5, zorder=2)
        ax.scatter(x+np.linspace(-.1,.1,len(vals)), vals, s=260, color=color, zorder=3,
                   edgecolor='#F4F8F1', linewidth=2)
    ax.set(xlim=(-.55,1.55), ylim=(.8,2.5), xticks=[0,1], yticks=[1,1.5,2,2.5])
    ax.set_xticklabels(['Low salinity\n(n = 3)','High salinity\n(n = 4)'], fontsize=22)
    clean(ax)
    ax.tick_params(axis='x', labelsize=22)
    ax.set_title('NPQ (dimensionless)\nHeat-dissipation index', fontsize=24, fontweight='bold', pad=18)
    ax.text(.5, -.42, 'Points: leaf means; bar: group mean\nmatched Q–Ci grid', ha='center', va='top',
            transform=ax.transAxes, fontsize=18, color=base.SECONDARY, linespacing=1.15)
    save(fig, 'npq')

def plots():
    method_plot(); capacity_plot(); surface_plot(); npq_plot()

def build():
    base.fonts(); areal_fonts(); plots()
    c=Canvas(str(TMP/'base.pdf'),pagesize=(W,H))
    c.setTitle('Forcing Plants to Participate')
    c.setAuthor('Riley Leff, Acadia Conboy, Keryn Gedan')
    c.setFillColor(HexColor('#F9FCF7')); c.rect(0,0,W,H,fill=1,stroke=0)
    # 300 ppi redraw of leaf-stoma-pencil.png, stretched by poster_warp
    # (scripts/redraw_stoma_background.py).
    c.drawImage(str(ASSETS/'leaf-stoma-pencil-300ppi.jpg'),0,0,W,H)
    fonts = {'PlexSans':'Areal-Regular', 'PlexSansBold':'Areal-Bold',
             'PlexSansBoldItalic':'Areal-BoldItalic', 'PlexSerif':'Areal-Regular',
             'PlexSerifBold':'Areal-Bold'}
    def text(s,x,y,size=32,font='PlexSans',color=base.INK,center=False):
        c.setFillColor(HexColor(color)); c.setFont(fonts[font],size)
        (c.drawCentredString if center else c.drawString)(x*72,H-y*72,s)
    def rich(parts,x,y,size,font='PlexSansBold',color=base.INK):
        """Centred line with subscripts: parts are str or (base, subscript)."""
        runs=[]
        for part in parts:
            if isinstance(part,str): runs.append((part,size,0))
            else: runs += [(part[0],size,0),(part[1],size*.68,-size*.2)]
        width=sum(pdfmetrics.stringWidth(s,fonts[font],sz) for s,sz,_ in runs)
        cx=x*72-width/2
        c.setFillColor(HexColor(color))
        for s,sz,rise in runs:
            c.setFont(fonts[font],sz); c.drawString(cx,H-y*72+rise,s)
            cx+=pdfmetrics.stringWidth(s,fonts[font],sz)
    text('Forcing Plants',24,2.1,150,'PlexSerifBold',center=True)
    text('to Participate',24,3.98,150,'PlexSerifBold',center=True)
    text('Riley Leff, Acadia Conboy, Keryn Gedan',24,5.0,44,center=True)
    text('Department of Biological Sciences, George Washington University',24,5.68,30,center=True)
    text('Morella cerifera',11.8,8.2,30,'PlexSansBoldItalic',center=True)
    text('Morella cerifera',36.3,8.2,30,'PlexSansBoldItalic',center=True)
    text('Low CO₂ forces plants to open their stomata.',11.8,16.45,28,'PlexSansBold',center=True)
    text('Quickly restoring CO₂ reveals carbon assimilation.',36.3,16.45,28,'PlexSansBold',center=True)
    # Overlays tied to the drawing follow the background's stretch.
    Y = lambda y: float(to_out(y))
    text("If a leaf isn't photosynthesizing,",24,Y(17.35),68,'PlexSerif', '#FFFFFF',True)
    text('is it incapable or unwilling?',24,Y(18.75),82,'PlexSerifBold','#FFFFFF',True)
    for i,line in enumerate(['Plants can close','their stomata to','cut off gas exchange.']):
        text(line,6.6,Y(20.0)+i*.47,28,'PlexSansBold',color=base.BLUE,center=True)
    def arrow(x1,y1,x2,y2,color):
        c.saveState(); c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color)); c.setLineWidth(7); c.setLineCap(1)
        x1,y1,x2,y2=x1*72,H-Y(y1)*72,x2*72,H-Y(y2)*72
        path=c.beginPath(); path.moveTo(x1,y1)
        path.curveTo(x1+(x2-x1)*.33,y1+(y2-y1)*.20+7,
                     x1+(x2-x1)*.7,y1+(y2-y1)*.8-3,x2,y2)
        c.drawPath(path)
        angle=math.atan2(y2-y1,x2-x1)
        for offset in [-.45,.45]:
            c.line(x2,y2,x2-27*math.cos(angle+offset),y2-27*math.sin(angle+offset))
        c.restoreState()
    arrow(7.7,19.65,10.9,18.9,base.BLUE)
    arrow(40.0,19.4,38.0,18.5,base.ORANGE)
    text('CO₂ in',42.0,Y(19.6),31,'PlexSansBold',color=base.ORANGE,center=True)
    arrow(37.0,19.25,40.0,20.7,base.BLUE)
    text('Water out',42.0,Y(20.9),31,'PlexSansBold',color=base.BLUE,center=True)
    text('Phragmites australis',12.6,24.25,30,'PlexSansBoldItalic',center=True)
    text('Phragmites australis',35,24.25,30,'PlexSansBoldItalic',center=True)
    rich(['Capacity estimates overlap across salinity levels, but carboxylation (',('V','cmax'),') is'],
         12,33.55,25)
    rich(['less than half that of a same-haplotype Gulf Coast population. ',('J','max'),' is similar.'],
         12,34.05,25)
    text('A representative photosynthetic',31.9,33.6,24,'PlexSansBold',center=True)
    text('response surface.',31.9,34.05,24,'PlexSansBold',center=True)
    text('Higher NPQ at higher salinity,',41.1,33.15,21,'PlexSansBold',center=True)
    text('where light availability',41.1,33.55,21,'PlexSansBold',center=True)
    text('is also higher.',41.1,33.95,21,'PlexSansBold',center=True)
    def logo(filename,x,y,w,h,valign='center'):
        path=ASSETS/filename
        if 'pencil' not in filename: path=knockout(path,enclosed=filename=='quadrat-labs.png')
        im=Image.open(path); iw,ih=im.size
        scale=min(w*72/iw,h*72/ih)
        dy={'center':(h*72-ih*scale)/2,'bottom':0}[valign]
        c.drawImage(str(path),x*72+(w*72-iw*scale)/2,H-(y+h)*72+dy,iw*scale,ih*scale,mask='auto')
    # Header, left: study system icons on a shared baseline, labelled.
    for filename,x,w,h,label,font in [
            ('morella-pencil.png',.6,3.05,4.6,'Morella cerifera','PlexSansBoldItalic'),
            ('phragmites-pencil.png',4.1,3.05,4.6,'Phragmites australis','PlexSansBoldItalic'),
            ('licor-6800-pencil.png',7.6,4.3,3.0,'LI-6800','PlexSansBold'),
            ('brownsville-forest-pencil.png',12.35,3.7,3.4,'Coastal forest','PlexSansBold')]:
        logo(filename,x,5.1-h,w,h,'bottom')
        text(label,x+w/2,5.75,23,font,center=True)
    # Header, right: a 4-column grid. Row 1: partner marks;
    # row 2: Astera wordmark across two columns, then the QR code and its caption.
    col=[31.7+i*3.95 for i in range(4)]
    # Box heights are optical: seals read smaller than a solid square of the same size.
    for i,(filename,h) in enumerate([('nsf.png',2.55),('vcr-lter.png',2.5),
                                      ('gw-biology-hand-drawn.jpg',2.2),('quadrat-labs.png',2.0)]):
        logo(filename,col[i]+.3,.6+(2.55-h)/2,3.35,h)
    logo('astera.png',col[0]+.3,3.75,7.3,1.55)
    url='https://rileyleff.com/blog/forcing-plants-to-participate'
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,border=0)
    qr.add_data(url); qr.make(fit=True)
    matrix=qr.get_matrix(); size=2.05*72; module=size/len(matrix)
    x0,y0=(col[2]+1.0)*72,H-5.75*72
    c.setFillColor(HexColor('#10271C'))
    for row,line in enumerate(matrix):
        for col_,dark in enumerate(line):
            if dark: c.rect(x0+col_*module,y0+size-(row+1)*module,module,module,fill=1,stroke=0)
    text('Read the story',col[3]+.35,4.55,26,'PlexSansBold')
    text('rileyleff.com',col[3]+.35,5.05,26,'PlexSans')
    c.linkURL(url,(x0,y0,x0+size,y0+size),relative=0)
    c.showPage(); c.save()
    writer=PdfWriter(); page=writer.add_page(PdfReader(TMP/'base.pdf').pages[0])
    for stem,x,y,w,h in [('method',1.5,8.6,45,7.2),('capacity',2.0,24.6,20,8.3),
                         ('surface',23.9,24.7,16,8.4),('npq',37.6,25.4,7.0,7.0)]:
        source=PdfReader(TMP/f'{stem}.pdf').pages[0]
        page.merge_transformed_page(source,Transformation().scale(w*72/float(source.mediabox.width)).translate(x*72,H-(y+h)*72))
    dest=base.PDF_OUT/'forcing-plants-to-participate-harlan-pencil-poster.pdf'
    writer.add_metadata({'/Title':'Forcing Plants to Participate','/Author':'Riley Leff, Acadia Conboy, Keryn Gedan'})
    writer.write(dest)
    print(dest)

if __name__=='__main__':
    build()
