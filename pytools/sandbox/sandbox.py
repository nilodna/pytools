import os

def crop_image(fname):
    # function to remove white spaces from pdf and png figures
    extension = fname.split('.')[-1]
    if extension == 'png':
        os.system('convert -trim %s %s'%(fname,fname))
    elif extension == 'pdf':
        os.system('pdfcrop %s %s'%(fname,fname))
