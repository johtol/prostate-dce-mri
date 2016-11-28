from mevis import *

import matplotlib.mlab as mlab
import numpy as np
import pylab as py
from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path
from mpldatacursor import HighlightingDataCursor, DataCursor
import matplotlib.pyplot as plt
import os
import timeit


#import matplolib.rc as rc



global CTX
global roi_dat
#global dat

# Use this to access the network context
CTX=ctx.owner().owner()
im = ctx.field("SubImage1.output0").image()
roi_dat = []

def initFigure(control):
    #initial figure

  figure = control.object().figure()
  figure.set_facecolor("#e0e0e0")
  figure.set_dpi(96)
  figure.set_size_inches(6,3)
  subplot = figure.add_subplot(111)
  subplot.hold(False)
  t = np.arange(0.0, 3.0, 0.01)
  s = np.sin(2*np.pi*t)
  subplot.plot(t, s)


def clearFigure():
  # clears the figure
  control = ctx.control("canvas").object()
  control.figure().clear()
  # clear the user interaction of plotC example:
  control.myUserInteraction = None

def applyroi():
  t = get_roi()
  #t = get_roi(im)

  #ctx.field("SubImage1.apply").touch()
  return

def get_roi():

#reads the image from the extent tou ROI
        #ctx.field("SubImage1.apply").touch()
        clearFigure()
        control = ctx.control("canvas").object()
        figure = control.figure()

        figure.clear()
        im = ctx.field("SubImage1.output0").image()

        start = timeit.default_timer()
        print "Start calculations...Please wait!!"
        #--------------------------------------------------------------------------------------------------------

        #print "x",ctx.field("SubImage1.x").value
        #print "sx",ctx.field("SubImage1.sx").value
        #print "y",ctx.field("SubImage1.y").value
        #print "sy",ctx.field("SubImage1.sy").value
        #print "z",ctx.field("SubImage1.z").value
        #print "sz",ctx.field("SubImage1.sz").value

        #list p contains the points included in the ROI the user draws.
        p = []
        xmin = ctx.field("SubImage1.x").value
        xmax = ctx.field("SubImage1.sx").value
        ymin = ctx.field("SubImage1.y").value
        ymax = ctx.field("SubImage1.sy").value
        zmin = ctx.field("SubImage1.z").value
        zmax = ctx.field("SubImage1.sz").value



        #print i,j,k
        for i in py.frange(xmin,xmax):
          for k in py.frange(ymin,ymax):
            for j in py.frange(zmax,zmin):
                offset = (i,k,j)
                p.append(offset)


        Anew = np.array(p)
        print "Anew is:", Anew
        print "Anew size:", Anew.size
        print "len(Anew):", len(Anew)

        #load the whole dataset in a numpy array
        dat = np.loadtxt(r"C:\Users\johnnie\PycharmProjects\Scatterplot\vtiparams.txt")
        #dat = np.loadtxt(r"C:\Users\johnnie\PycharmProjects\Scatterplot\vtiparams.txt")
        print "shape Anew:", Anew.shape
        print "dat shape:", dat.shape
        print "len(dat):", len(dat)


        #code comparing (x,y,z) triplets of the selected roi with the whole dataset dat.
        #for indexA in xrange(0, Anew.shape[0]):
        #    for indexB in xrange(0, dat.shape[0]):
        #        if (Anew[indexA, :] == dat[indexB, 0:3]).all():
        #            roi_dat.append(dat[indexB, :])

        roi_datX = []
        for indexX in range(0, len(dat)):

            if (dat[indexX, 0] <= xmax) and (dat[indexX, 0] >= xmin):
                roi_datX.append(dat[indexX, :])

        roiX = np.array(roi_datX)
        print "roiX.shape", roiX.shape

        roi_datY = []
        for indexY in range(0, len(roiX)):
            if (roiX[indexY, 1] <= ymax) and (roiX[indexY, 1] >= ymin):
                roi_datY.append(roiX[indexY, :])
        roiY = np.array(roi_datY)
        print "roiY.shape", roiY.shape

        #global list containing the parameters values for the selected ROI.
        global roi_dat

        for indexZ in range(0, len(roiY)):
            if (roiY[indexZ, 2] <= zmin) and (roiY[indexZ, 2] >= zmax):
                roi_dat.append(roiY[indexZ, :])
        roi_all = np.array(roi_dat)
        print "roi all",roi_all.shape
        goodrows = [row for row in roi_all if not(row[3] == 0 and row[4] == 0 and row[5] == 0 and row[6] == 0).all()]
        roi = np.array(goodrows)
        #print roi
        print "roi.shape", roi.shape
        #print roiZ

        stop = timeit.default_timer()
        print "processing time:", stop - start



        subplots=[]
        subplot1 = figure.add_subplot(441)
        subplot2 = figure.add_subplot(445)
        subplot3 = figure.add_subplot(446)
        subplot4 = figure.add_subplot(449)
        subplot5 = figure.add_subplot(4,4,10)
        subplot6 = figure.add_subplot(4,4,11)
        subplot7 = figure.add_subplot(4,4,13)
        subplot8 = figure.add_subplot(4,4,14)
        subplot9 = figure.add_subplot(4,4,15)
        subplot10 = figure.add_subplot(4,4,16)



        x, y = roi[:, 3], roi[:, 4]  #ke,ktrans
        x1, y1 = roi[:, 5], roi[:, 6]  # ve,vplasma
        a = np.array([x,y])  #ke,ktrans
        a = a.transpose()

        b = np.array([x,y1])  #ke,vplasma
        b = b.transpose()

        c = np.array([x,x1])  #ke,ve
        c = c.transpose()

        d = np.array([y,x1])  #ve,ktrans
        d = d.transpose()

        e = np.array([x1,y1])  #ve,vplasma
        e = e.transpose()

        f = np.array([y,y1])  #ktrans, vplasma
        f = f.transpose()

        data = []

        data0 = [Datum(*xy) for xy in a]   #ke,ktrans
        data.append(data0)
        data1 = [Datum(*xy) for xy in b]   #ke,vplasma
        data.append(data1)
        data2 = [Datum(*xy) for xy in c]   #ke,ve
        data.append(data2)
        data3 = [Datum(*xy) for xy in d]   #ve,ktrans
        data.append(data3)
        data4 = [Datum(*xy) for xy in e]   #ve,vplasma
        data.append(data4)
        data5 = [Datum(*xy) for xy in f]   #ktrans, vplasma
        data.append(data5)

              #print data
              #print len(data)
        fig, axes = plt.subplots(ncols=4, nrows=4)
        #lman = []
        #for ax in axes.flat:
        #  for i,j in axes[i][:], axes[:][j]:

        #kep vs ktrans
        subplot2.plot(x, y, 'bo',  ls='',  picker=3)
        subplots.append(subplot2)
              #subplot2.set_xlabel('Ke')
        subplot2.set_ylabel('Ktrans')
        subplot2.set_xlim((min(x)-50, max(x)+50))
              #subplot2.set_xticklabels([])
        subplot2.set_ylim((min(y)-50, max(y)+50))
        lman1 = LassoManager(subplot2, data[0])
        print "lman1 is", id(lman1)

        # kep vs ktrans
        axes[1][0].plot(x, y, 'bo',  ls='',  picker=3)
        axes[1][0].set_xlabel('Ke')
        axes[1][0].set_ylabel('Ktrans')
        axes[1][0].set_xlim((min(x)-50, max(x)+50))
        axes[1][0].set_ylim((min(y)-50, max(y)+50))
        lman1 = LassoManager(axes[1][0], data[0])
              #print "lman1 is", id(lman1)

        #  kep vs vplasma
        subplot7.plot(x, y1, 'bo',  ls='',  picker=3)
        subplot7.set_xlabel('Ke')
        subplot7.set_ylabel('Vplasma')
        subplot7.set_xlim((min(x)-50, max(x)+50))
        subplot7.set_ylim((min(y)-50, max(y)+50))
        lman1.add_axis(axes[3][0], data[1])
        subplots.append(subplot7)
              #print "lman1 is", id(lman1)
        #kep vs vplasma
        axes[3][0].plot(x, y1, 'bo',  ls='',  picker=3)
        axes[3][0].set_xlabel('Ke')
        axes[3][0].set_ylabel('Vplasma')
        axes[3][0].set_xlim((min(x)-50, max(x)+50))
        axes[3][0].set_ylim((min(y1)-40, max(y1)+50))
        lman1.add_axis(axes[3][0], data[1])
             # print "lman2 is", id(lman2)

        #kep vs ve
        subplot4.plot(x, x1, 'bo',  ls='',  picker=3)
        #subplot4.set_xlabel('Ke')
        subplot4.set_ylabel('Ve')
        #subplot4.set_xlim((min(x)-50, max(x)+50))
        subplot4.set_ylim((min(x1)-40, max(x1)+50))
        lman1.add_axis(subplot4, data[2])
        subplots.append(subplot4)

        #kep vs ve
        axes[2][0].plot(x, x1, 'bo',  ls='',  picker=3)
        axes[2][0].set_xlabel('Ke')
        axes[2][0].set_ylabel('Ve')
        axes[2][0].set_xlim((min(x)-50, max(x)+50))
        axes[2][0].set_ylim((min(x1)-40, max(x1)+50))
        lman1.add_axis(axes[2][0], data[2])
              # print "lman3 is", id(lman3)

        #ve vs vplasma
        subplot9.plot(x1, y1, 'bo',  ls='',  picker=3)
        subplot9.set_xlabel('Ve')
         #subplot9.set_ylabel('Vplasma')
        subplot9.set_xlim((min(x1)-50, max(x1)+50))
        subplot9.set_ylim((min(y1)-40, max(y1)+50))
        subplot9.set_yticklabels([])
        lman1.add_axis(subplot9, data[4])
        subplots.append(subplot9)

        #ve vs vplasma
        axes[3][2].plot(x1, y1, 'bo',  ls='',  picker=3)
        axes[3][2].set_xlabel('Ve')
        axes[3][2].set_ylabel('Vplasma')
        axes[3][2].set_xlim((min(x1)-50, max(x1)+50))
        axes[3][2].set_ylim((min(y1)-40, max(y1)+50))
        lman1.add_axis(axes[3][2], data[4])
             # print "lman4 is", id(lman4)


        #ktrans vs ve
        subplot5.plot(y, x1, 'bo',  ls='',  picker=3)
          #subplot5.set_xlabel('Ktrans')
          #subplot5.set_ylabel('Ve')
          #subplot5.set_xlim((min(y)-50, max(y)+50))
          #subplot5.set_ylim((min(x1)-40, max(x1)+50))
        subplot5.set_yticklabels([])
        lman1.add_axis(subplot5, data[3])
        subplots.append(subplot5)

        #ktrans vs ve
        axes[2][1].plot(y, x1, 'bo',  ls='',  picker=3)
        axes[2][1].set_xlabel('Ktrans')
        axes[2][1].set_ylabel('Ve')
        axes[2][1].set_xlim((min(y)-50, max(y)+50))
        axes[2][1].set_ylim((min(x1)-40, max(x1)+50))
        lman1.add_axis(axes[2][1], data[3])
         # print "lman5 is", id(lman5)


        #ktrans vs vplasma
        subplot8.plot(y, y1, 'bo',  ls='',  picker=3)
        subplot8.set_xlabel('Ktrans')
        #subplot8.set_ylabel('Vplasma')
        subplot8.set_xlim((min(y)-50, max(y)+50))
        subplot8.set_ylim((min(y1)-40, max(y1)+50))
        subplot8.set_yticklabels([])
        lman1.add_axis(subplot8, data[5])
        subplots.append(subplot8)

        #ktrans vs vplasma
        axes[3][1].plot(y, y1, 'bo',  ls='',  picker=3)
        axes[3][1].set_xlabel('Ktrans')
        axes[3][1].set_ylabel('Vplasma')
        axes[3][1].set_xlim((min(y)-50, max(y)+50))
        axes[3][1].set_ylim((min(y1)-40, max(y1)+50))
        lman1.add_axis(axes[3][1], data[5])
             # print "lman6 is", id(lman6)

        #non-visible subplots
        axes[0][0].plot(x,x)
        axes[0][0].set_visible(False)
        axes[1][1].plot(y,y)
        axes[1][1].set_visible(False)
        axes[2][2].plot(x1,x1)
        axes[2][2].set_visible(False)
        axes[3][3].plot(y1,y1)
        axes[3][3].set_visible(False)
        axes[0][1].plot(x,y)
        axes[0][1].set_visible(False)
        axes[0][2].plot(x,y)
        axes[0][2].set_visible(False)
        axes[0][3].plot(x,y)
        axes[0][3].set_visible(False)
        axes[1][2].plot(x,y)
        axes[1][2].set_visible(False)
        axes[1][3].plot(x,y)
        axes[1][3].set_visible(False)
        axes[2][3].plot(x,y)
        axes[2][3].set_visible(False)
        IndexedHighlight(axes.flat)
        IndexedHighlight(subplots)



        #removes rows which have zero values for all the PK parameters
        #goodrows = [row for row in roi if not(row[3] == 0 and row[4] == 0 and row[5] == 0 and row[6] == 0).all()]
        #roi_nozero = np.array(goodrows)
        #print "dat_nozero",roi_nozero.shape
        #
        #roi_X, roi_Y = roi_nozero[:, 3], roi_nozero[:, 4]  #ke,ktrans
        #roi_X1, roi_Y1 = roi_nozero[:, 5], roi_nozero[:, 6]  # ve,vplasma

        #removes rows which have zero values for all the PK parameters
        goodrows1 = [row for row in roi if not(row[3] == 0 ).all()]
        dat_nozero_X = np.array(goodrows1)
        #print dat_nozero_X.shape

        goodrows2 = [row for row in roi if not(row[4] == 0 ).all()]
        dat_nozero_X1 = np.array(goodrows2)

        goodrows3 = [row for row in roi if not(row[5] == 0 ).all()]
        dat_nozero_Y = np.array(goodrows3)

        goodrows4 = [row for row in roi if not(row[6] == 0).all()]
        dat_nozero_Y1 = np.array(goodrows4)
        #print "dat_nozero",roi_nozero.shape
        #
        #roi_X, roi_Y = roi_nozero[:, 3], roi_nozero[:, 4]  #ke,ktrans
        #roi_X1, roi_Y1 = roi_nozero[:, 5], roi_nozero[:, 6]  # ve,vplasma

        #hist ke
        mu1 = dat_nozero_X[:,3].mean()
        sigma1 = dat_nozero_X[:,3].std()
        #fit1 = mu1 + sigma1*dat[:,3]
        #axes[0][0] = fig.add_subplot(4,4,-15)
        #hist1 = mu1 + sigma1*x
        n, bins, patches = subplot1.hist(dat_nozero_X[:,3], 30, normed=1, facecolor='green', alpha=0.75)
        # add a 'best fit' line
        bs1 = mlab.normpdf( bins, mu1, sigma1)
        subplot1.plot(bins, bs1, 'r--', linewidth=1)
        subplot1.set_xlabel('Ke')
        #subplot1.set_xlim([0,max(dat_nozero_X[:,3])])
        #subplot1.set_ylim([0,max(dat_nozero_X[:,3])])
        #axes[0][0].set_visible(True)

        #hist ke
        #mu1 = dat[:,3].mean()
        #sigma1 = dat[:,3].std()
        #axes[0][0] = fig.add_subplot(4,4,-15)
        #    #hist1 = mu1 + sigma1*x
        #n,bins,patches = py.hist(x, 50, normed=1, histtype='stepfilled')
        #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
        #exp1 = py.normpdf(bins, mu1, sigma1)
        #axes[0][0].plot(bins,exp1,'k--',linewidth=1.5)
        #axes[0][0].set_visible(True)

        #hist ktrans
        mu2 = dat_nozero_X1[:,4].mean()
        sigma2 = dat_nozero_X1[:,4].std()
        #fit2 = mu2 + sigma2*dat[:,4]
        #axes[1][1] = fig.add_subplot(4,4,-10)
        n, bins, patches = subplot3.hist(dat_nozero_X1[:,4], 30, normed=1, facecolor='green', alpha=0.75)
        bs2 = mlab.normpdf( bins, mu2, sigma2)
        l = subplot3.plot(bins, bs2, 'r--', linewidth=1)
        #subplot3.set_xlim([0,max(dat_nozero_X1[:,4])])
        #subplot3.set_ylim([0,max(dat_nozero_X1[:,4])])
        subplot3.set_xticklabels([])
        subplot3.set_yticklabels([])
        #axes[1][1].set_visible(True)

        #hist ktrans
        #mu2 = dat[:,4].mean()
        #sigma2 = dat[:,4].std()
        #axes[1][1] = fig.add_subplot(4,4,-10)
        #n,bins,patches = py.hist(y, 50, normed=1, histtype='stepfilled')
        #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
        #exp2 = py.normpdf(bins, mu2, sigma2)
        #axes[1][1].plot(bins,exp2,'k--',linewidth=1.5)
        #axes[1][1].set_visible(True)

        #hist ve
        mu3 = dat_nozero_Y[:,5].mean()
        sigma3 = dat_nozero_Y[:,5].std()
        #fit3 = mu3 + sigma3*dat[:,5]
        #axes[2][2] = fig.add_subplot(4,4,-5)
        n, bins, patches = subplot6.hist(dat_nozero_Y[:,5], 30, normed=1, facecolor='green', alpha=0.75)
        bs3 = mlab.normpdf( bins, mu3, sigma3)
        l = subplot6.plot(bins, bs3, 'r--', linewidth=1)
        #subplot6.set_xlim([0,max(dat_nozero_Y[:,5])])
        subplot6.set_ylim([0,100])
        subplot6.set_xticklabels([])
        subplot6.set_yticklabels([])
        #axes[2][2].set_visible(True)

        #hist ve
        #mu3 = dat[:,5].mean()
        #sigma3 = dat[:,5].std()
        #axes[2][2] = fig.add_subplot(4,4,-5)
        #n,bins,patches = py.hist(x1, 50, normed=1, histtype='stepfilled')
        #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
        #exp3 = py.normpdf(bins, mu3, sigma3)
        #axes[2][2].plot(bins,exp3,'k--',linewidth=1.5)
        #axes[2][2].set_visible(True)

        #hist vplasma
        mu4 = dat_nozero_Y1[:,6].mean()
        sigma4 = dat_nozero_Y1[:,6].std()
        #fit4 = mu4 + sigma4*dat[:,6]
        #axes[3][3] = fig.add_subplot(4, 4, 0)
        n, bins, patches = subplot10.hist(dat_nozero_Y1[:,6], 30, normed=1, facecolor='green', alpha=0.75)
        bs4 = mlab.normpdf( bins, mu4, sigma4)
        l = subplot10.plot(bins, bs4, 'r--', linewidth=1)
        subplot10.set_xlabel('Vplasma')
        #subplot10.set_xlim([0,max(dat_nozero_Y1[:,5])])
        #subplot10.set_ylim([0,max(dat_nozero_Y1[:,5])])
        subplot10.set_yticklabels([])
        #subplot.set_visible(True)

        #hist vplasma
        #mu4 = dat[:,6].mean()
        #sigma4 = dat[:,6].std()
        #axes[3][3] = fig.add_subplot(4, 4, 0)
        #n,bins,patches = py.hist(y1, 50, normed=1, histtype='stepfilled')
        #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
        #exp4 = py.normpdf(bins, mu4, sigma4)
        #axes[3][3].plot(bins,exp4,'k--',linewidth=1.5)
        #axes[3][3].set_visible(True)


        plt.subplots_adjust(left=0.1, right=0.95, wspace=0.8, hspace=0.9)
        figure.canvas.draw()
        plt.draw()
        return roi_dat
        #print array.size, len(array)
          #a = np.array(a,dtype="float")
          #print array




def showLutEditorPanel():
  ctx.showWindow("LutEditor")
  return


def showRawDataPanel():
  ctx.showWindow("ChooseADataSet")
  return

def deletemarkers(event):
  if event["key"] == "Delete CSO":
    ctx.field("CSOManager3.numSelectedCSOs").value = 0
  return


def expandDirectoryName():
  exp = ctx.expandFilename(ctx.field("Raw_Data.source").stringValue())
  dir = MLABFileDialog.getExistingDirectory(exp, "Open directory", MLABFileDialog.ShowDirsOnly)
  if dir:
    ctx.field("Raw_Data.source").value = ctx.unexpandFilename(dir)



def selectFirstTab():
  ctx.control("Raw_Data.tabViewOfDDI").selectTabAtIndex(0)

def clearLogAndImport():
  ctx.field("Raw_Data.clearConsole").touch()
  ctx.field("Raw_Data.dplImport").touch()


def changePK():

  PKparams = ctx.field("PKparams").value

  if PKparams == "None":
    # Reset current value
    global gLock
    gLock = True
    ctx.field("FieldIterator.curIndex").setIntValue(-1)
    ctx.field("FieldIterator.curValue").setStringValue("")
    gLock = False
    MLAB.log("Please select a PK parameter.")

  elif PKparams == "Kep":
    #ctx.field("PK_params_data.dplImport")
    ctx.field("FieldIterator.curIndex").value = 2
    ctx.field("FieldIterator.curValue").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5KEP/"
    ctx.field("PK_params_data.source").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5KEP/"
    MLAB.log("Kep param selected.")
    #print 'select a parameter please'

  elif PKparams == "Ktrans":

    #ctx.field("PK_params_data.dplImport").value =""
    ctx.field("FieldIterator.curIndex").value = 3
    ctx.field("FieldIterator.curValue").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5KTRANS/"
    ctx.field("PK_params_data.source").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5KTRANS/"
    MLAB.log("Ktrans param selected.")

  elif PKparams == "Ve":
    #ctx.field("PK_params_data.dplImport").value=""
    ctx.field("FieldIterator.curIndex").value = 4
    ctx.field("FieldIterator.curValue").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5VE/"
    ctx.field("PK_params_data.source").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5VE/"
    MLAB.log("Ve param selected.")

  elif PKparams == "Vplasma":
    #ctx.field("PK_params_data.dplImport").value =""
    ctx.field("FieldIterator.curIndex").value = 5
    ctx.field("FieldIterator.curValue").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5VPLASMA/"
    ctx.field("PK_params_data.source").value = "C:/Users/johnnie/Documents/MeVis/ImageData/NKI/quant/tDCE-2.5VPLASMA/"
    #ctx.field("PK_params_data.dplImport").value = ""
    MLAB.log("Vplasma param selected.")
  return

#def setmarker():
#   #controls = [ctx.field("Raw Data")  ]
#   #for c in controls:
#   #    c.setCheckableButtons(checkable)
#   SetMarker = ctx.field("SetMarker").value
#   if SetMarker== "No":
#        MLAB.log("You can't set a marker")
#
#   else:
#
#       ctx.field("View2D3.self")
#       MLAB.log("You can set a marker")
#
def deletemarkers():
  #if event["key"] == "Delete CSO":
    ctx.field("SoView2DMarkerEditor0.deleteAll").touch()
    #ctx.field("CSOManager3.csoRemoveSelected").touch()
    return

def deletecsos():
  #if event["key"] == "Delete CSO":
    ctx.field("CSOManager3.removeAllCSOsAndGroups").touch()
    #ctx.field("CSOManager3.csoRemoveSelected").touch()
    return

def changeLayout():
  layoutValue = ctx.field("Layout").value

  if layoutValue == "Axial":
    ctx.field("OrthoView2D1.layout").value = "LAYOUT_AXIAL"
  elif layoutValue == "Sagittal":
    ctx.field("OrthoView2D1.layout").value = "LAYOUT_SAGITTAL"
  elif layoutValue == "Coronal":
    ctx.field("OrthoView2D1.layout").value = "LAYOUT_CORONAL"
  elif layoutValue == "Row Equal":
    ctx.field("OrthoView2D1.layout").value = "LAYOUT_ROW_EQUAL"



def changeLayout2():
  layoutValue = ctx.field("Layout2").value

  if layoutValue == "Axial":
    ctx.field("OrthoView2D2.layout").value = "LAYOUT_AXIAL"
  elif layoutValue == "Sagittal":
    ctx.field("OrthoView2D2.layout").value = "LAYOUT_SAGITTAL"
  elif layoutValue == "Coronal":
    ctx.field("OrthoView2D2.layout").value = "LAYOUT_CORONAL"
  #elif layoutValue == "Cube":
  #  ctx.field("OrthoView2D2.layout").value = "LAYOUT_CUBE"
  #elif layoutValue == "Cube Equal":
   # ctx.field("OrthoView2D2.layout").value = "LAYOUT_CUBE_EQUAL"
 # elif layoutValue == "Cube Customized":
   # ctx.field("OrthoView2D2.layout").value = "LAYOUT_CUBE_CUSTOMIZED"
 # elif layoutValue == "Row":
 #   ctx.field("OrthoView2D2.layout").value = "LAYOUT_ROW"
  elif layoutValue == "Row Equal":
    ctx.field("OrthoView2D2.layout").value = "LAYOUT_ROW_EQUAL"
  #elif layoutValue == "Row Axialextra":
   # ctx.field("OrthoView2D2.layout").value = "LAYOUT_ROW_AXIALEXTRA"
 # elif layoutValue == "Column":
  #  ctx.field("OrthoView2D2.layout").value = "LAYOUT_COLUMN"
 # elif layoutValue == "Column Equal":
   # ctx.field("OrthoView2D2.layout").value = "LAYOUT_COLUMN_EQUAL"



#  """
#Show how to use a lasso to select a set of points and get the indices
#of the selected points.  A callback is used to change the color of the
#selected points

##epitrepei kai tis dyo leitourgies alla oxi se ola ta subplot. I highlighting kanonika, enw i lasso mono ston axes katw dexia.


class Datum(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('blue')
    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:self.color = self.colorin
        else:self.color = self.colorout


class LassoManager(object):
    #class for highlighting region of points within a Lasso

    def __init__(self, ax, data):


        # self.highlights = IndexedHighlight(HighlightingDataCursor)
        self.axes = [ax]
        self.canvas = ax.figure.canvas
        self.data = [data]

        self.Nxy = [len(data)]

        facecolors = [d.color for d in data]
        self.xys = [[(d.x, d.y) for d in data]]
        fig = ax.figure
        self.collection = [RegularPolyCollection(
            fig.dpi, 6, sizes=(1,),
            facecolors=facecolors,
            offsets = self.xys[0],
            transOffset = ax.transData)]

        ax.add_collection(self.collection[0])

        # self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        #prosthiki gia metatropi sxediasmoy se shift+lmb
        self.cidpress = self.canvas.mpl_connect('key_press_event', self.onpress)
        # if self.event == 'key_press_event':
        #
        #   del self.highlights
        # if self.cidpress:

    def callback(self, verts):

        LassoManager.has_been_called = True

        axind = self.axes.index(self.current_axes)
        facecolors = self.collection[axind].get_facecolors()
        print "The id of this lasso is", id(self)


        p = path.Path(verts)
        ind = p.contains_points(self.xys[axind])

        #print ind prints boolean array of points in subplot where true means that the point is included
        if IndexedHighlight.has_been_called == True:
            # print hilight
            # if hilight<> None:
            #     hilight.set_visible(False)

            for i in range(len(self.xys[axind])):

                         if ind[i]:

                            # hilight.hide()
                            #  facecolors[i] = Datum.colorin
                            #     artists = [axind.lines[0] for ax in axes]


                            axes[1][0].plot(x[i], y[i], 'ro',  ls='',  picker=3)
                            axes[3][0].plot(x[i], y1[i], 'ro',  ls='',  picker=3)
                            axes[2][0].plot(x[i], x1[i], 'ro',  ls='',  picker=3)
                            axes[2][1].plot(y[i], x1[i], 'ro',  ls='',  picker=3)
                            axes[3][2].plot(x1[i], y1[i], 'ro',  ls='',  picker=3)
                            axes[3][1].plot(y[i], y1[i], 'ro',  ls='',  picker=3)

                         else:
                            # DataCursor.hide(self)
                            # facecolors[i] = Datum.colorout
                            # hilight.hide()
                            axes[1][0].plot(x[i], y[i], 'bo',  ls='',  picker=3)
                            axes[3][0].plot(x[i], y1[i], 'bo',  ls='',  picker=3)
                            axes[2][0].plot(x[i], x1[i], 'bo',  ls='',  picker=3)
                            axes[2][1].plot(y[i], x1[i], 'bo',  ls='',  picker=3)
                            axes[3][2].plot(x1[i], y1[i], 'bo',  ls='',  picker=3)
                            axes[3][1].plot(y[i], y1[i], 'bo',  ls='',  picker=3)

            plt.draw()


            IndexedHighlight.has_been_called = False
        else:


                    # z.highlight.set_visible(False)
                    # z.annotations.set_visible(False)
                    for i in range(len(self.xys[axind])):

                         if ind[i]:
                            # facecolors[i] = Datum.colorin
                            axes[1][0].plot(x[i], y[i], 'ro',  ls='',  picker=3)
                            axes[3][0].plot(x[i], y1[i], 'ro',  ls='',  picker=3)
                            axes[2][0].plot(x[i], x1[i], 'ro',  ls='',  picker=3)
                            axes[2][1].plot(y[i], x1[i], 'ro',  ls='',  picker=3)
                            axes[3][2].plot(x1[i], y1[i], 'ro',  ls='',  picker=3)
                            axes[3][1].plot(y[i], y1[i], 'ro',  ls='',  picker=3)

                         else:
                            # facecolors[i] = Datum.colorout
                            axes[1][0].plot(x[i], y[i], 'bo',  ls='',  picker=3)
                            axes[3][0].plot(x[i], y1[i], 'bo',  ls='',  picker=3)
                            axes[2][0].plot(x[i], x1[i], 'bo',  ls='',  picker=3)
                            axes[2][1].plot(y[i], x1[i], 'bo',  ls='',  picker=3)
                            axes[3][2].plot(x1[i], y1[i], 'bo',  ls='',  picker=3)
                            axes[3][1].plot(y[i], y1[i], 'bo',  ls='',  picker=3)

                    plt.draw()

        # del lman1, lman2, lman3, lman4, lman5

        # print (self.xys) #prints all the points in the axes
        # for i in range(len(self.xys)):
        #     if ind[i]:
        #         # print "i", i
        #         facecolors[i] = Datum.colorin
        #         print (ind[i]) #prints the positions of the points that are in the lasso in the boolean array, e.g. True
        #         print i,a[i]
        #     else:
        #         facecolors[i] = Datum.colorout
        #         # print (facecolors[i])
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso
        # noinspection PyArgumentList

    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        self.current_axes = event.inaxes
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        print "onpress"

        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def add_axis(self, ax, data):
        # adds the axis, collections, etc information to the LassoManager, so that when that specific axis is clicked
        # the Lasso uses the data for that axis.
        self.axes.append(ax)
        self.data.append(data)

        self.Nxy.append( len(data) )

        facecolors = [d.color for d in data]
        self.xys.append( [(d.x, d.y) for d in data] )
        fig = ax.figure
        self.collection.append( RegularPolyCollection(
            fig.dpi, 6, sizes=(1,),
            facecolors=facecolors,
            offsets = self.xys[-1],
            transOffset = ax.transData))

        ax.add_collection(self.collection[-1])


LassoManager.has_been_called = False


class IndexedHighlight(HighlightingDataCursor):
    #class for highlighting single points
    # DataCursor.visible=False

    def __init__(self, axes, **kwargs ):

        # Use the first plotted Line2D in each axes
        artists = [ax.lines[0] for ax in axes]
        # print artists
        kwargs['display'] = 'single'
        HighlightingDataCursor.__init__(self, artists, **kwargs)
        self.highlights = [self.create_highlight(artist) for artist in artists]
        plt.setp(self.highlights, visible=False)


    def update(self, event, annotation):
        IndexedHighlight.has_been_called = True

        # Hide all other annotations
        plt.setp(self.highlights, visible=False)


        # Highlight everything with the same index.
        artist, ind = event.artist, event.ind

        global roi_dat
        roi = np.array(roi_dat)
        self.roi = roi
        #dat = np.loadtxt(r"C:\Users\johnnie\Desktop\vtiparams.txt")
        #self.dat = dat
        for original, highlight in zip(self.artists, self.highlights):
            x, y = original.get_data()

            highlight.set(visible=True, xdata=x[ind], ydata=y[ind])

        #anatref = self.dat[ind][:,0:3]
        anatref = self.roi[ind][:,0:3]
        ctx.field("WorldVoxelConvert4.voxelPos").value = [float(self.roi[:, 0][ind]), float(self.roi[:, 1][ind]), float(self.roi[:, 2][ind])]
        ctx.field("pos.editingOn").value="True"
        ctx.field("pos.drawingOn").value="True"
        ctx.field("pos.drawEditingRect").value="True"
        #ctx.field("pos.cooperative").value = "True"
        #ctx.field("pos.createNewMode").value = "True"
        #ctx.field("pos.worldPosition").value=[float(self.dat[:, 0][ind]), float(self.dat[:, 1][ind]), float(self.dat[:, 2][ind])]

        ctx.field("SoView2DPosition3.editingOn").value="True"
        ctx.field("SoView2DPosition3.drawingOn").value="True"
        ctx.field("SoView2DPosition3.drawEditingRect").value="True"

        #prints the number of the selected points in the scatterplots
        print len(anatref)
        print '(x,y,z):', self.roi[:, 0][ind], self.roi[:, 1][ind], self.roi[:, 2][ind]


        if LassoManager.has_been_called == True:
                #roi = self.roi
                #dat = self.dat
                #x, y = dat[:, 3], dat[:, 4]  #ke,ktrans
                #x1, y1 = dat[:, 5], dat[:, 6]  # ve,vplasma
                x, y = roi[:, 3], roi[:, 4]  #ke,ktrans
                x1, y1 = roi[:, 5], roi[:, 6]  # ve,vplasma

                axes[1][0].plot(x, y, 'bo',  ls='',  picker=3)
                axes[3][0].plot(x, y1, 'bo',  ls='',  picker=3)
                axes[2][0].plot(x, x1, 'bo',  ls='',  picker=3)
                axes[2][1].plot(y, x1, 'bo',  ls='',  picker=3)
                axes[3][2].plot(x1, y1, 'bo',  ls='',  picker=3)
                axes[3][1].plot(y, y1, 'bo',  ls='',  picker=3)
                # LassoManager.has_been_called=False
                plt.draw()
                DataCursor.update(self, event, annotation)
                LassoManager.has_been_called = False
        else:

                DataCursor.update(self, event, annotation)

IndexedHighlight.has_been_called = False



def main():

    clearFigure()

    control = ctx.control("canvas").object()
    figure = control.figure()

    figure.clear()


    # Pick up the image
    #im = ctx.field("SubImage1.output0").image()
    ##
    #if im:
    #   get_roi()

    #get_roi(im)




    #font = {'family' : 'normal',
    #    'weight' : 'bold',
    #    'size'   : 10}
    #
    #rc('font', **font)

    #fig = control.figure()


    #path = MLABFileManager.getTmpDir()
    #file = "C:\\Users\\johnnie\\PycharmProjects\\Scatterplot\\vtiparams_small.txt".format(path)
    #f = open(file, "r")
    #dat = []
    #for line in open(file):
    #  #print line
    #  #f.write ("I am almost done\n")
    #  dat.append(line)
    #print dat

    #   print dat
    #   print dat[:,1]


    #f.close()


    #----
    #dat = np.loadtxt(r"C:\Users\johnnie\PycharmProjects\Scatterplot\vtiparams.txt")
    #dat = np.loadtxt(r"C:\Users\johnnie\PycharmProjects\Scatterplot\vtiparams_small.txt")
    #----


    #dat = np.loadtxt(r"C:\Users\johnnie\Desktop\vtiparams.txt")
    ##
    ##    #
    subplots=[]
    subplot1 = figure.add_subplot(441)
    subplot2 = figure.add_subplot(445)
    subplot3 = figure.add_subplot(446)
    subplot4 = figure.add_subplot(449)
    subplot5 = figure.add_subplot(4,4,10)
    subplot6 = figure.add_subplot(4,4,11)
    subplot7 = figure.add_subplot(4,4,13)
    subplot8 = figure.add_subplot(4,4,14)
    subplot9 = figure.add_subplot(4,4,15)
    subplot10 = figure.add_subplot(4,4,16)




    x, y = dat[:, 3], dat[:, 4]  #ke,ktrans
    x1, y1 = dat[:, 5], dat[:, 6]  # ve,vplasma
    a = np.array([x,y])  #ke,ktrans
    a = a.transpose()

    b = np.array([x,y1])  #ke,vplasma
    b = b.transpose()

    c = np.array([x,x1])  #ke,ve
    c = c.transpose()

    d = np.array([y,x1])  #ve,ktrans
    d = d.transpose()

    e = np.array([x1,y1])  #ve,vplasma
    e = e.transpose()

    f = np.array([y,y1])  #ktrans, vplasma
    f = f.transpose()

    data = []

    data0 = [Datum(*xy) for xy in a]   #ke,ktrans
    data.append(data0)
    data1 = [Datum(*xy) for xy in b]   #ke,vplasma
    data.append(data1)
    data2 = [Datum(*xy) for xy in c]   #ke,ve
    data.append(data2)
    data3 = [Datum(*xy) for xy in d]   #ve,ktrans
    data.append(data3)
    data4 = [Datum(*xy) for xy in e]   #ve,vplasma
    data.append(data4)
    data5 = [Datum(*xy) for xy in f]   #ktrans, vplasma
    data.append(data5)

        #print data
        #print len(data)
    fig, axes = plt.subplots(ncols=4, nrows=4)
        # lman = []
        # for ax in axes.flat:
        # for i,j in axes[i][:], axes[:]j]:

        #kep vs ktrans
    subplot2.plot(x, y, 'bo',  ls='',  picker=3)
    subplots.append(subplot2)
       #subplot2.set_xlabel('Ke')
    subplot2.set_ylabel('Ktrans')
    subplot2.set_xlim((min(x)-50, max(x)+50))
    #subplot2.set_xticklabels([])
    subplot2.set_ylim((min(y)-50, max(y)+50))
    lman1 = LassoManager(subplot2, data[0])
    print "lman1 is", id(lman1)
    #kep vs ktrans
    axes[1][0].plot(x, y, 'bo',  ls='',  picker=3)
    axes[1][0].set_xlabel('Ke')
    axes[1][0].set_ylabel('Ktrans')
    axes[1][0].set_xlim((min(x)-50, max(x)+50))
    axes[1][0].set_ylim((min(y)-50, max(y)+50))
    lman1 = LassoManager(axes[1][0], data[0])
        #print "lman1 is", id(lman1)

    #kep vs vplasma
    subplot7.plot(x, y1, 'bo',  ls='',  picker=3)
    subplot7.set_xlabel('Ke')
    subplot7.set_ylabel('Vplasma')
    subplot7.set_xlim((min(x)-50, max(x)+50))
    subplot7.set_ylim((min(y)-50, max(y)+50))
    lman1.add_axis(axes[3][0], data[1])
    subplots.append(subplot7)
        #print "lman1 is", id(lman1)
        #kep vs vplasma
    axes[3][0].plot(x, y1, 'bo',  ls='',  picker=3)
    axes[3][0].set_xlabel('Ke')
    axes[3][0].set_ylabel('Vplasma')
    axes[3][0].set_xlim((min(x)-50, max(x)+50))
    axes[3][0].set_ylim((min(y1)-40, max(y1)+50))
    lman1.add_axis(axes[3][0], data[1])
        # print "lman2 is", id(lman2)

    #kep vs ve
    subplot4.plot(x, x1, 'bo',  ls='',  picker=3)
    #subplot4.set_xlabel('Ke')
    subplot4.set_ylabel('Ve')
    #subplot4.set_xlim((min(x)-50, max(x)+50))
    subplot4.set_ylim((min(x1)-40, max(x1)+50))
    lman1.add_axis(subplot4, data[2])
    subplots.append(subplot4)
    #kep vs ve
    axes[2][0].plot(x, x1, 'bo',  ls='',  picker=3)
    axes[2][0].set_xlabel('Ke')
    axes[2][0].set_ylabel('Ve')
    axes[2][0].set_xlim((min(x)-50, max(x)+50))
    axes[2][0].set_ylim((min(x1)-40, max(x1)+50))
    lman1.add_axis(axes[2][0], data[2])
        # print "lman3 is", id(lman3)

    #ve vs vplasma
    subplot9.plot(x1, y1, 'bo',  ls='',  picker=3)
    subplot9.set_xlabel('Ve')
    #subplot9.set_ylabel('Vplasma')
    subplot9.set_xlim((min(x1)-50, max(x1)+50))
    subplot9.set_ylim((min(y1)-40, max(y1)+50))
    subplot9.set_yticklabels([])
    lman1.add_axis(subplot9, data[4])
    subplots.append(subplot9)
    #ve vs vplasma
    axes[3][2].plot(x1, y1, 'bo',  ls='',  picker=3)
    axes[3][2].set_xlabel('Ve')
    axes[3][2].set_ylabel('Vplasma')
    axes[3][2].set_xlim((min(x1)-50, max(x1)+50))
    axes[3][2].set_ylim((min(y1)-40, max(y1)+50))
    lman1.add_axis(axes[3][2], data[4])
        # print "lman4 is", id(lman4)


    #ktrans vs ve
    subplot5.plot(y, x1, 'bo',  ls='',  picker=3)
    #subplot5.set_xlabel('Ktrans')
    #subplot5.set_ylabel('Ve')
    #subplot5.set_xlim((min(y)-50, max(y)+50))
    #subplot5.set_ylim((min(x1)-40, max(x1)+50))
    subplot5.set_yticklabels([])
    lman1.add_axis(subplot5, data[3])
    subplots.append(subplot5)
    #ktrans vs ve
    axes[2][1].plot(y, x1, 'bo',  ls='',  picker=3)
    axes[2][1].set_xlabel('Ktrans')
    axes[2][1].set_ylabel('Ve')
    axes[2][1].set_xlim((min(y)-50, max(y)+50))
    axes[2][1].set_ylim((min(x1)-40, max(x1)+50))
    lman1.add_axis(axes[2][1], data[3])
    # print "lman5 is", id(lman5)


    #ktrans vs vplasma
    subplot8.plot(y, y1, 'bo',  ls='',  picker=3)
    subplot8.set_xlabel('Ktrans')
    #subplot8.set_ylabel('Vplasma')
    subplot8.set_xlim((min(y)-50, max(y)+50))
    subplot8.set_ylim((min(y1)-40, max(y1)+50))
    subplot8.set_yticklabels([])
    lman1.add_axis(subplot8, data[5])
    subplots.append(subplot8)
    #ktrans vs vplasma
    axes[3][1].plot(y, y1, 'bo',  ls='',  picker=3)
    axes[3][1].set_xlabel('Ktrans')
    axes[3][1].set_ylabel('Vplasma')
    axes[3][1].set_xlim((min(y)-50, max(y)+50))
    axes[3][1].set_ylim((min(y1)-40, max(y1)+50))
    lman1.add_axis(axes[3][1], data[5])
        # print "lman6 is", id(lman6)

    #non-visible subplots
    axes[0][0].plot(x,x)
    axes[0][0].set_visible(False)
    axes[1][1].plot(y,y)
    axes[1][1].set_visible(False)
    axes[2][2].plot(x1,x1)
    axes[2][2].set_visible(False)
    axes[3][3].plot(y1,y1)
    axes[3][3].set_visible(False)
    axes[0][1].plot(x,y)
    axes[0][1].set_visible(False)
    axes[0][2].plot(x,y)
    axes[0][2].set_visible(False)
    axes[0][3].plot(x,y)
    axes[0][3].set_visible(False)
    axes[1][2].plot(x,y)
    axes[1][2].set_visible(False)
    axes[1][3].plot(x,y)
    axes[1][3].set_visible(False)
    axes[2][3].plot(x,y)
    axes[2][3].set_visible(False)
    IndexedHighlight(axes.flat)
    IndexedHighlight(subplots)

    #removes rows which have zero values for all the PK parameters
    goodrows1 = [row for row in dat if not(row[3] == 0 ).all()]
    dat_nozero_X = np.array(goodrows1)
    #print dat_nozero_X.shape

    goodrows2 = [row for row in dat if not(row[4] == 0 ).all()]
    dat_nozero_X1 = np.array(goodrows2)

    goodrows3 = [row for row in dat if not(row[5] == 0 ).all()]
    dat_nozero_Y = np.array(goodrows3)

    goodrows4 = [row for row in dat if not(row[6] == 0).all()]
    dat_nozero_Y1 = np.array(goodrows4)
    #print "dat_nozero",roi_nozero.shape
    #
    #roi_X, roi_Y = roi_nozero[:, 3], roi_nozero[:, 4]  #ke,ktrans
    #roi_X1, roi_Y1 = roi_nozero[:, 5], roi_nozero[:, 6]  # ve,vplasma

    #hist ke
    mu1 = dat_nozero_X[:,3].mean()
    sigma1 = dat_nozero_X[:,3].std()
    #fit1 = mu1 + sigma1*dat[:,3]
    #axes[0][0] = fig.add_subplot(4,4,-15)
    #hist1 = mu1 + sigma1*x
    n, bins, patches = subplot1.hist(dat_nozero_X[:,3], 30, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    bs1 = mlab.normpdf( bins, mu1, sigma1)
    l = subplot1.plot(bins, bs1, 'r--', linewidth=1)
    subplot1.set_xlabel('Ke')
    #axes[0][0].set_visible(True)

    #hist ke
    #mu1 = dat[:,3].mean()
    #sigma1 = dat[:,3].std()
    #axes[0][0] = fig.add_subplot(4,4,-15)
    #    #hist1 = mu1 + sigma1*x
    #n,bins,patches = py.hist(x, 50, normed=1, histtype='stepfilled')
    #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    #exp1 = py.normpdf(bins, mu1, sigma1)
    #axes[0][0].plot(bins,exp1,'k--',linewidth=1.5)
    #axes[0][0].set_visible(True)

    #hist ktrans
    mu2 = dat_nozero_X1[:,4].mean()
    sigma2 = dat_nozero_X1[:,4].std()
    #fit2 = mu2 + sigma2*dat[:,4]
    #axes[1][1] = fig.add_subplot(4,4,-10)
    n, bins, patches = subplot3.hist(dat_nozero_X1[:,4], 30, normed=1, facecolor='green', alpha=0.75)
    bs2 = mlab.normpdf( bins, mu2, sigma2)
    l = subplot3.plot(bins, bs2, 'r--', linewidth=1)
    subplot3.set_xticklabels([])
    subplot3.set_yticklabels([])
    #axes[1][1].set_visible(True)

    #hist ktrans
    #mu2 = dat[:,4].mean()
    #sigma2 = dat[:,4].std()
    #axes[1][1] = fig.add_subplot(4,4,-10)
    #n,bins,patches = py.hist(y, 50, normed=1, histtype='stepfilled')
    #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    #exp2 = py.normpdf(bins, mu2, sigma2)
    #axes[1][1].plot(bins,exp2,'k--',linewidth=1.5)
    #axes[1][1].set_visible(True)

    #hist ve
    mu3 = dat_nozero_Y[:,5].mean()
    sigma3 = dat_nozero_Y[:,5].std()
    #fit3 = mu3 + sigma3*dat[:,5]
    #axes[2][2] = fig.add_subplot(4,4,-5)
    n, bins, patches = subplot6.hist(dat_nozero_Y[:,5], 30, normed=1, facecolor='green', alpha=0.75)
    bs3 = mlab.normpdf( bins, mu3, sigma3)
    l = subplot6.plot(bins, bs3, 'r--', linewidth=1)
    subplot6.set_xticklabels([])
    subplot6.set_yticklabels([])
    #axes[2][2].set_visible(True)

    #hist ve
    #mu3 = dat[:,5].mean()
    #sigma3 = dat[:,5].std()
    #axes[2][2] = fig.add_subplot(4,4,-5)
    #n,bins,patches = py.hist(x1, 50, normed=1, histtype='stepfilled')
    #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    #exp3 = py.normpdf(bins, mu3, sigma3)
    #axes[2][2].plot(bins,exp3,'k--',linewidth=1.5)
    #axes[2][2].set_visible(True)

    #hist vplasma
    mu4 = dat_nozero_Y1[:,6].mean()
    sigma4 = dat_nozero_Y1[:,6].std()
    #fit4 = mu4 + sigma4*dat[:,6]
    #axes[3][3] = fig.add_subplot(4, 4, 0)
    n, bins, patches = subplot10.hist(dat_nozero_Y1[:,6], 30, normed=1, facecolor='green', alpha=0.75)
    bs4 = mlab.normpdf( bins, mu4, sigma4)
    l = subplot10.plot(bins, bs4, 'r--', linewidth=1)
    subplot10.set_xlabel('Vplasma')
    subplot10.set_yticklabels([])
    #subplot.set_visible(True)

    #hist vplasma
    #mu4 = dat[:,6].mean()
    #sigma4 = dat[:,6].std()
    #axes[3][3] = fig.add_subplot(4, 4, 0)
    #n,bins,patches = py.hist(y1, 50, normed=1, histtype='stepfilled')
    #py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    #exp4 = py.normpdf(bins, mu4, sigma4)
    #axes[3][3].plot(bins,exp4,'k--',linewidth=1.5)
    #axes[3][3].set_visible(True)


    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.6, hspace=0.7)



    #plt.show()

    plt.draw()


print "exec"









