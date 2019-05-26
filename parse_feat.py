from xml.dom import minidom
import glob
import pandas as pd

def parse():
    name_top=[]
    xmin_top=[]
    ymax_top=[]
    width_top=[]
    height_top=[]
    l=glob.glob("/home/siva/faster-rcnn.pytorch/VOC2012/VOC2012/Annotations/*.xml")
    df_cols=['name','x','y','w','h']
    out_df = pd.DataFrame(columns = df_cols)
    for k in range (len(l)):
        name=[]
        xmin=[]
        ymax=[]
        width=[]
        height=[]
        doc=minidom.parse(l[k])
        itemlist = doc.getElementsByTagName('name')
        v=doc.getElementsByTagName('bndbox')
        for i in range(len(itemlist)):
            if v[i].parentNode=='object' and itemlist[i].parentNode=='object':
                name.append(itemlist[i].childNodes[0].nodeValue)
                xmin.append(v[i].childNodes[1].childNodes[0].nodeValue)
                ymax.append(v[i].childNodes[7].childNodes[0].nodeValue)
                xmin=v[i].childNodes[1].childNodes[0].nodeValue
                ymin=v[i].childNodes[3].childNodes[0].nodeValue
                xmax=v[i].childNodes[5].childNodes[0].nodeValue
                ymax=v[i].childNodes[7].childNodes[0].nodeValue
                width.append(xmax-xmin)
                height.append(ymax-ymin)

        name_top.append(name)
        xmin_top.append(xmin)
        ymax_top.append(ymax)
        width_top.append(width)
        height_top.append(height)
        out_df = out_df.append(pd.Series([name_top, xmin_top, ymax_top, width_top,height_top], 
                                         index = df_cols), 
                               ignore_index = True)

        return out_df
