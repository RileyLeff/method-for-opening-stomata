"""Print poster: generated botanical background, real data, vector labels."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import qrcode
import math
from fontTools.ttLib import TTFont as FontToolsFont
from fontTools.varLib.instancer import instantiateVariableFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import build_harlan_2026 as base

ROOT = base.ROOT
TMP = base.TMP / 'leaf'
TMP.mkdir(exist_ok=True)
ASSETS = base.OUT / 'assets'
W, H = 3456, 2592

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
                         'ytick.color':'#243B30', 'mathtext.fontset':'custom',
                         'mathtext.rm':'Areal Poster', 'mathtext.it':'Areal Poster:italic',
                         'mathtext.bf':'Areal Poster:bold', 'mathtext.sf':'Areal Poster'})

def save(fig, name):
    fig.savefig(TMP / f'{name}.pdf', transparent=True)
    plt.close(fig)

def plots():
    data = json.loads((base.ASSETS / 'low-co2-opening.json').read_text())
    df = pd.DataFrame(data['records'])
    fig, axes = plt.subplots(1, 2, figsize=(45, 5.8))
    fig.subplots_adjust(left=.075, right=.95, bottom=.24, top=.81, wspace=.50)
    for ax, col, color, label, ylim in zip(axes, ['gsw', 'a'], [base.BLUE, base.CYAN],
        [r'$g_{sw}\;(\mathrm{mol\,m^{-2}\,s^{-1}})$'+'\nStomatal openness',
         r'$A\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nPhotosynthetic rate'], [(-.015,.24),(0,17)]):
        for x in [30,60,90]:
            ax.axvline(x, color=base.SECONDARY, lw=1.5, ls='--', alpha=.5)
        ax.plot(df.minute, df[col], color=color, lw=5.5)
        ax.set(xlim=(0,120), ylim=ylim, xticks=[0,30,60,90,120])
        ax.set_xlabel(r'$t\;(\mathrm{min})$, Elapsed time', fontsize=27, labelpad=10)
        ax.set_ylabel(label, fontsize=27, labelpad=14)
        ax.tick_params(labelsize=28, pad=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        if col in ('gsw', 'a'):
            co2 = ax.twinx()
            co2.plot(df.minute, df.ca, color=base.ORANGE, lw=4.5, ls=(0,(7,4)))
            co2.set_ylim(0,500 if col == 'gsw' else 600)
            co2.set_yticks([125,275,420])
            co2.set_ylabel(r'$C_a\;(\mathrm{\mu mol\,mol^{-1}})$'+'\nChamber CO₂',color=base.ORANGE,fontsize=25,labelpad=12)
            co2.tick_params(axis='y',colors=base.ORANGE,labelsize=26,pad=8)
            co2.spines[['top','left','bottom']].set_visible(False)
            co2.spines['right'].set_color(base.ORANGE)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y',colors=color)
        if col == 'a':
            initial = df.loc[df.minute.ge(25) & df.minute.lt(30), 'a'].mean()
            peak = df.loc[df.minute.ge(90) & df.minute.le(95), 'a'].max()
            for value, name, x in [(initial,'Initial 420 ppm',.035),(peak,'Return peak',.53)]:
                ax.axhline(value,color=base.SECONDARY,lw=2,ls=(0,(3,4)),alpha=.85)
                ax.text(x,value+.45,name,transform=ax.get_yaxis_transform(),
                        color=base.INK,fontsize=22,va='bottom')
    save(fig, 'method')
    # Re-use the actual posterior calculation, with a compact poster layout.
    import xarray as xr
    from scipy.stats import gaussian_kde
    tree=xr.open_datatree(ROOT/'outputs/phrag_brownsville_tier_c_posterior.nc')
    draws,_=base.build_draws_and_summary(tree)
    fig,axes=plt.subplots(1,2,figsize=(20,5.5))
    fig.subplots_adjust(left=.075,right=.97,top=.64,bottom=.32,wspace=.30)
    for ax,p,limit,title,ref in zip(axes,['Vcmax25','Jmax25'],[120,220],['Carbon fixation','Electron transport'],[84,116]):
        xx=np.linspace(0,limit,600)
        for zone,color in [('High',base.BLUE),('Low',base.ORANGE)]:
            yy=gaussian_kde(draws[p][zone])(xx)
            ax.fill_between(xx,0,yy,color=color,alpha=.22)
            ax.plot(xx,yy,color=color,lw=4)
        ax.axvline(ref,color=base.INK,lw=2,ls=(0,(4,4)))
        ax.set_xlim(0,limit); ax.set_ylim(bottom=0); ax.set_yticks([])
        ax.annotate('Gulf Coast\n(Nguyen 2013)',xy=(ref,ax.get_ylim()[1]*.9),
                    xytext=(.79,1.01),textcoords='axes fraction',ha='center',va='bottom',
                    fontsize=22,arrowprops={'arrowstyle':'-','lw':2,'color':base.INK})
        units = r' ($\mu$mol m$^{-2}$ s$^{-1}$)' if p=='Vcmax25' else r' ($\mu$mol e$^{-}$ m$^{-2}$ s$^{-1}$)'
        ax.set_xlabel((r'$V_{cmax,25}$' if p=='Vcmax25' else r'$J_{max,25}$')+units+'\n'+title+' capacity',fontsize=25,labelpad=11)
        ax.tick_params(labelsize=27)
        ax.spines[['top','right','left']].set_visible(False)
    save(fig,'capacity'); tree.close()
    data=json.loads((base.ASSETS/'phrag-p10-surface.json').read_text())
    ci,q=np.meshgrid(data['ci'],data['q'])
    fig=plt.figure(figsize=(17,7.2))
    ax=fig.add_subplot(projection='3d',computed_zorder=False)
    ax.plot_surface(ci,q,np.array(data['assimilation']),cmap='viridis',vmin=0,vmax=26,rcount=len(data['q']),ccount=len(data['ci']),linewidth=0,antialiased=True,rasterized=True)
    ax.set(xlim=(0,max(data['ci'])),ylim=(0,2000),zlim=(0,27),xticks=[0,600,1200],yticks=[0,1000,2000],zticks=[0,10,20])
    ax.set_xlabel(r'$C_i\;(\mathrm{\mu mol\,mol^{-1}})$'+'\nInternal CO₂',fontsize=21,labelpad=20)
    ax.set_ylabel(r'$Q\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nIncident light',fontsize=21,labelpad=23)
    ax.set_zlabel(r'$A\;(\mathrm{\mu mol\,m^{-2}\,s^{-1}})$'+'\nPhotosynthetic rate',fontsize=21,labelpad=17)
    ax.tick_params(labelsize=21,pad=2)
    ax.view_init(elev=24,azim=-135)
    ax.set_box_aspect((1.35,1.15,.65),zoom=.97)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_pane_color((.95,.98,.95,.1))
        axis._axinfo['grid']['color']=(.3,.4,.35,.16)
    fig.subplots_adjust(left=0,right=1,bottom=.09,top=1)
    save(fig,'surface')
    leaves=pd.read_csv(ROOT/'outputs/phrag_brownsville_npq_overall_leaf_means.csv')
    fig,ax=plt.subplots(figsize=(6.6,5.3))
    fig.subplots_adjust(left=.20,right=.94,bottom=.30,top=.71)
    for x,zone,color in [(0,'High',base.BLUE),(1,'Low',base.ORANGE)]:
        vals=leaves.loc[leaves.zone.eq(zone),'NPQ'].to_numpy()
        ax.scatter(x+np.linspace(-.09,.09,len(vals)),vals,s=150,color=color,zorder=3)
        ax.hlines(vals.mean(),x-.20,x+.20,color=color,lw=4)
    ax.set(xlim=(-.5,1.5),ylim=(.8,2.5),xticks=[0,1],yticks=[1,1.5,2,2.5])
    ax.set_xticklabels(['Low salinity\n(n = 3)','High salinity\n(n = 4)'],fontsize=20)
    ax.tick_params(axis='y',labelsize=23)
    ax.set_title('NPQ (dimensionless)\nHeat-dissipation index',fontsize=23,fontweight='bold',pad=16)
    ax.text(.5,-.59,'Leaf means; matched Q–Ci grid',ha='center',transform=ax.transAxes,fontsize=17)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    save(fig,'npq')

def build():
    base.fonts(); areal_fonts(); plots()
    c=Canvas(str(TMP/'base.pdf'),pagesize=(W,H))
    c.setTitle('Forcing Plants to Participate')
    c.setAuthor('Riley Leff, Acadia Conboy, Keryn Gedan')
    c.setFillColor(HexColor('#F9FCF7')); c.rect(0,0,W,H,fill=1,stroke=0)
    # 300 ppi redraw of leaf-stoma-pencil.png (scripts/redraw_stoma_background.py).
    c.drawImage(str(ASSETS/'leaf-stoma-pencil-300ppi.jpg'),0,0,W,H)
    def text(s,x,y,size=32,font='PlexSans',color=base.INK,center=False):
        font = {'PlexSans':'Areal-Regular', 'PlexSansBold':'Areal-Bold',
                'PlexSansBoldItalic':'Areal-BoldItalic', 'PlexSerif':'Areal-Regular',
                'PlexSerifBold':'Areal-Bold'}[font]
        c.setFillColor(HexColor(color)); c.setFont(font,size)
        (c.drawCentredString if center else c.drawString)(x*72,H-y*72,s)
    text('Forcing Plants to Participate',24,2.5,112,'PlexSerifBold',center=True)
    text('Riley Leff, Acadia Conboy, Keryn Gedan',24,3.65,39,center=True)
    text('Department of Biological Sciences, George Washington University',24,4.4,27,center=True)
    text('Morella cerifera',11.8,7.55,30,'PlexSansBoldItalic',center=True)
    text('Morella cerifera',36.3,7.55,30,'PlexSansBoldItalic',center=True)
    text('Low CO₂ forces plants to open their stomata.',11.8,13.95,28,'PlexSansBold',center=True)
    text('Quickly restoring CO₂ reveals carbon assimilation.',36.3,13.95,28,'PlexSansBold',center=True)
    text("If a leaf isn't photosynthesizing,",24,17.35,68,'PlexSerif', '#FFFFFF',True)
    text('is it incapable or unwilling?',24,18.75,82,'PlexSerifBold','#FFFFFF',True)
    for i,line in enumerate(['Plants can close','their stomata to','cut off gas exchange.']):
        text(line,7.6,20.0+i*.47,28,'PlexSansBold',color=base.BLUE,center=True)
    def arrow(x1,y1,x2,y2,color):
        c.saveState(); c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color)); c.setLineWidth(7); c.setLineCap(1)
        x1,y1,x2,y2=x1*72,H-y1*72,x2*72,H-y2*72
        path=c.beginPath(); path.moveTo(x1,y1)
        path.curveTo(x1+(x2-x1)*.33,y1+(y2-y1)*.20+7,
                     x1+(x2-x1)*.7,y1+(y2-y1)*.8-3,x2,y2)
        c.drawPath(path)
        angle=math.atan2(y2-y1,x2-x1)
        for offset in [-.45,.45]:
            c.line(x2,y2,x2-27*math.cos(angle+offset),y2-27*math.sin(angle+offset))
        c.restoreState()
    arrow(8.7,19.65,10.9,18.9,base.BLUE)
    arrow(40.0,19.4,38.0,18.5,base.ORANGE)
    text('CO₂ in',42.0,19.6,31,'PlexSansBold',color=base.ORANGE,center=True)
    arrow(37.0,19.25,40.0,20.7,base.BLUE)
    text('Water out',42.0,20.9,31,'PlexSansBold',color=base.BLUE,center=True)
    text('Phragmites australis',12,22.25,30,'PlexSansBoldItalic',center=True)
    text('Phragmites australis',35,22.25,30,'PlexSansBoldItalic',center=True)
    text('Low salinity (n = 3)',3.5,23.12,26,'PlexSansBold',base.BLUE)
    text('High salinity (n = 5)',12.2,23.12,26,'PlexSansBold',base.ORANGE)
    text('Capacity estimates overlap across salinity levels.',12,29.25,26,'PlexSansBold',center=True)
    text('A representative photosynthetic',32.0,29.05,24,'PlexSansBold',center=True)
    text('response surface.',32.0,29.52,24,'PlexSansBold',center=True)
    text('Higher NPQ at higher salinity,',40.7,28.75,21,'PlexSansBold',center=True)
    text('where light availability',40.7,29.15,21,'PlexSansBold',center=True)
    text('is also higher.',40.7,29.55,21,'PlexSansBold',center=True)
    def logo(filename,x,y,w,h):
        path=ASSETS/filename
        im=Image.open(path); iw,ih=im.size
        scale=min(w*72/iw,h*72/ih)
        c.drawImage(str(path),x*72+(w*72-iw*scale)/2,H-(y+h)*72+(h*72-ih*scale)/2,iw*scale,ih*scale,mask='auto')
    # Mirrored shoulder pairs share a centerline and equal center-to-center spacing.
    logo('licor-6800-pencil.png',1.6,.8,4.6,4.4)
    logo('brownsville-forest-pencil.png',6.8,.8,4.6,4.4)
    logo('morella-pencil.png',36.6,.8,4.6,4.4)
    logo('phragmites-pencil.png',41.8,.8,4.6,4.4)
    logo('gw-biology-hand-drawn.jpg',7.7,32.6,2.6,2.45)
    logo('nsf.png',13.8,32.62,2.4,2.4)
    text('NSF GRFP',15.0,35.45,21,'PlexSansBold',center=True)
    logo('astera.png',18.25,33.1,5.5,1.5)
    logo('quadrat-labs.png',25.65,32.6,2.7,2.45)
    text('Quadrat Labs',27.0,35.45,21,'PlexSansBold',center=True)
    logo('vcr-lter.png',31.75,32.6,2.5,2.45)
    url='https://rileyleff.com/blog/forcing-plants-to-participate'
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,border=4)
    qr.add_data(url); qr.make(fit=True)
    matrix=qr.get_matrix(); size=2.4*72; module=size/len(matrix)
    x0,y0=37.8*72,H-35.02*72
    c.setFillColor(HexColor('#FFFFFF')); c.rect(x0,y0,size,size,fill=1,stroke=0)
    c.setFillColor(HexColor('#10271C'))
    for row,line in enumerate(matrix):
        for col,dark in enumerate(line):
            if dark: c.rect(x0+col*module,y0+size-(row+1)*module,module,module,fill=1,stroke=0)
    text('Read The Story on',39,35.40,20,'PlexSansBold',center=True)
    text('rileyleff.com',39,35.78,20,'PlexSansBold',center=True)
    c.linkURL(url,(x0,y0,x0+size,y0+size),relative=0)
    c.showPage(); c.save()
    writer=PdfWriter(); page=writer.add_page(PdfReader(TMP/'base.pdf').pages[0])
    for stem,x,y,w,h in [('method',1.5,7.75,45,5.8),('capacity',2.0,23.4,20,5.5),('surface',23.7,21.9,17,7.2),('npq',37.4,23.1,6.6,5.3)]:
        source=PdfReader(TMP/f'{stem}.pdf').pages[0]
        page.merge_transformed_page(source,Transformation().scale(w*72/float(source.mediabox.width)).translate(x*72,H-(y+h)*72))
    dest=base.PDF_OUT/'forcing-plants-to-participate-harlan-pencil-poster.pdf'
    writer.add_metadata({'/Title':'Forcing Plants to Participate','/Author':'Riley Leff, Acadia Conboy, Keryn Gedan'})
    writer.write(dest)
    print(dest)

if __name__=='__main__':
    build()
