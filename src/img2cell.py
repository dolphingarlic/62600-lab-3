# %%
from phidl import Device
import math
import phidl.geometry as pg
import cv2
import numpy as np
import solarcellS23
import img2cell

##########################################
# Layers:
# 1 = contact opening etch
# 2 = aluminum metal wiring
# 3 = dummy fill pattern
# 11 = active device area
##########################################
# WRITE ON MLA
# Mask 1 = Layer 1
# Mask 2 = Layer 2 OR 3
# Ignore Layer 11, it's for visualization
##########################################
#
# IMPORTANT: Flatten design in KLayout
# and remove all hierarchy references before
# sending to MLA.  MLA code stalls otherwise
#
##########################################


### Building blocks for our solar cell ###

def outline(mc):
    """
    Creates the outline of the solar cell die.
    """
    D = pg.basic_die(
        size=(mc.width, mc.height),  # Size of die
        street_width=50,   # Width of corner marks for die-sawing
        street_length=5000,  # Length of corner marks for die-sawing
        layer=2,
        draw_bbox=False
    )
    return D


def solarpad(mc):
    """
    Creates the metal contact pad at the top of the solar cell.
    """
    P = Device()
    P << pg.rectangle(
        size=(mc.width - 2 * mc.spacing, mc.pad_size),
        layer=2
    ).move((mc.spacing, -mc.pad_size - mc.spacing))
    P << pg.rectangle(
        size=(mc.width - 2 * (mc.spacing + mc.contact_buffer),
              mc.pad_size - 2 * mc.contact_buffer),
        layer=1
    ).move((mc.spacing + mc.contact_buffer,
            -mc.pad_size - mc.spacing + mc.contact_buffer))
    P.move((-mc.width / 2, mc.height / 2))
    return P


def imagewire(mc, file_name, pixel_width=100, pixel_height=100):
    """
    Converts an image into metal wires to collect current from the P-N junction.
    The image should have an aspect ratio of `mc.device_width` by `mc.device_height`
    or else it will get stretched.
    """
    print(f'Converting {file_name} to GDS file..')

    # Step 1: Load image -> set size -> turn grayscale -> reduce dynamic range
    img = cv2.resize(
        cv2.imread(file_name),
        dsize=(mc.device_width // pixel_width,
               mc.device_height // pixel_height)
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #THIS IS DYNAMIC RANGE
    #max you could ever add is 127.5 if you divide 2
    gray = np.add(np.divide(gray, mc.dr_expand), mc.dr_shift) 
    if mc.noise_no == False:
        if type(mc.noise_gen) == bool:
            random_gray_i = np.random.choice(np.shape(gray)[0], 135, replace=False)
            random_gray_j = np.random.choice(np.shape(gray)[1], 160, replace=False)
            coords = zip(random_gray_i, random_gray_j)
            for coord in coords:
                gray[coord] = mc.noise_color
            mc.noise_gen = np.copy(gray)
        else:
            gray = np.copy(mc.noise_gen)
        cv2.imwrite(f'{file_name}{mc.noise_color}.bmp', gray)
    # Output grayscale image for testing purposes
    #cv2.imwrite(f'{file_name}{mc.dr_shift}.bmp', gray)

    # Step 2: Draw pixels on the device
    total_area = mc.device_width * (mc.device_height - mc.pad_size)
    used_area = 0
    y0 = (mc.device_height - mc.pad_size) / 2 - pixel_height
    x0 = -mc.device_width / 2
    W = Device()
    for i in range(gray.shape[1]):
        for j in range(gray.shape[0]):
            sz = max(mc.contact_buffer * 2,
                     gray[j, i] / 255 * pixel_height - 10)
            used_area += (sz + mc.contact_buffer)**2
            #size=(sz,x)
            if mc.rect_no:
                mc.rect_size = sz
            W << pg.rectangle(size=(sz, mc.rect_size), layer=1).move( 
                (x0 + i * pixel_width + (pixel_width - sz) / 2,
                 y0 - j * pixel_height + (pixel_height - mc.rect_size) / 2)
            )
            W << pg.rectangle(
                size=(sz + mc.contact_buffer * 2, mc.rect_size + mc.contact_buffer * 2), layer=2
            ).move(
                (x0 + i * pixel_width + (pixel_width - sz) / 2 - mc.contact_buffer,
                 y0 - j * pixel_height + (pixel_height - mc.rect_size) / 2 - mc.contact_buffer)
            )
    print(f'Pixel coverage: {used_area / total_area * 100}%')

    # Step 3: Connect the pixels to the top pad
    for i in range(gray.shape[1]):
        W << pg.rectangle(
            size=(mc.contact_buffer * 2, mc.device_height - pixel_height / 2),
            layer=2
        ).move(
            (x0 + i * pixel_width + pixel_width / 2 - mc.contact_buffer,
             -(mc.device_height + mc.pad_size - pixel_height) / 2)
        )
    return W

### The actual solar cell ###


def solarcell(mc, filename):
    """
    Put all the parts of the solar cell together.
    """
    D = Device('solarcell')
    # D << outline(mysolar)
    D << solarpad(mysolar)
    D << imagewire(mysolar, filename)
    # add label into cleaving street, as backup
    D << pg.text(
        text=mysolar.name,
        size=((mysolar.dicing_width - mc.dummy_buffer) * 2),
        justify='center',
        layer=2,
        font="Arial"
    ).move((0, -mysolar.height / 2 - mysolar.dicing_width + mc.dummy_buffer))

    # the next part is optional:
    if mc.usedummy:
        # add dummy squares to block the light
        DFiller = Device()
        DFiller << pg.rectangle(
            size=(mysolar.width - 2 * mysolar.dicing_width,
                  mysolar.height - 2 * mysolar.dicing_width),
            layer=10
        ).move(
            (-mysolar.width / 2 + mysolar.dicing_width,
             -mysolar.height / 2 + mysolar.dicing_width)
        )
        # run boolean operation to cut out the dummy blocks where there's a device...
        DBlock = Device()
        DBlock << pg.text(
            text=mysolar.name,
            size=mysolar.text_size,
            justify='center',
            layer=11,
            font="Arial"
        ).move((0, mysolar.height / 2 - mysolar.spacing + mysolar.text_size / 2))
        DBlock << pg.rectangle(
            size=(mc.device_width + mc.dummy_buffer * 2,
                  mc.device_height + mc.pad_size + mc.dummy_buffer * 2),
            layer=11
        ).move(
            (-mc.device_width / 2 - mc.dummy_buffer,
             -(mc.device_height + mc.pad_size) / 2 - mc.dummy_buffer)
        )
        for x in range(mysolar.width // mysolar.dummy_size):
            DBlock << pg.rectangle(
                size=(mysolar.dummy_gap, mysolar.height), layer=11
            ).move((-mysolar.width / 2 + x * mysolar.dummy_size, -mysolar.height / 2))
        for y in range(mysolar.height // mysolar.dummy_size):
            DBlock << pg.rectangle(
                size=(mysolar.width, mysolar.dummy_gap), layer=11
            ).move((-mysolar.width / 2, -mysolar.height / 2 + y * mysolar.dummy_size))
        D.add_ref(pg.boolean(A=DFiller, B=pg.union(DBlock, layer=11),
                  operation='A-B', precision=1e-6, layer=3))
        D << pg.rectangle(
            size=(mc.device_width + mc.dummy_buffer * 2,
                  mc.device_height + mc.pad_size + mc.dummy_buffer * 2),
            layer=11
        ).move(
            (-mc.device_width / 2 - mc.dummy_buffer,
             -(mc.device_height + mc.pad_size)/2 - mc.dummy_buffer)
        )
    else:
        D << pg.text(
            text=mysolar.name,
            size=mysolar.text_size,
            justify='center',
            layer=3,
            font="Arial"
        ).move((0, mysolar.height / 2 - mysolar.spacing + mysolar.text_size / 2))

    return D


# solar cell parameters, use object to store parameters:
class EmptyClass:
    pass


mysolar = EmptyClass()
# total size of each die
mysolar.width = 20000
mysolar.height = 20000
# the actual cell is inside the die, with a spacing. the contact pad height needs to be large enough to probe it
mysolar.spacing = 2000
mysolar.pad_size = 2500
# we don't want to protect against misalignment, so there's a mc.contact_buffer for the contact opening inside the metal shape
mysolar.contact_buffer = 6
# cell labeling (put your name here or something idk)
mysolar.name = 'andiqu and fatemaz'
mysolar.text_size = 200
# to have a more precise idea of the actual area, we'll shade the area around the cell.
# it'll be done with metal dummy squares, so accidental contact to a square will not risk short circuiting near the cleave line
mysolar.usedummy = True
mysolar.dummy_gap = 5
mysolar.dummy_size = 2000
mysolar.dummy_buffer = 10  # how close to put the dummy fillers to the actual device
# leave edges free from fillers to make it easier to aim when cleaving
mysolar.dicing_width = 125
#shifter variables
mysolar.dr_shift = 0
mysolar.rect_size = 100
mysolar.rect_no = True
mysolar.noise_no = True
mysolar.noise_color = 0
mysolar.noise_gen = False
mysolar.dr_expand = 2
# pre-calculate the actual device area
mysolar.device_height = mysolar.height - 2 * mysolar.spacing - mysolar.pad_size
mysolar.device_width = mysolar.width - 2 * mysolar.spacing


# generate the cells: explore different shifting minimums for same DR:
aDR = [0, 25, 50, 75, 100, 120]
# generate the cells: explore different dr ranges:
aDR_range = [1, 1.5, 2, 2.5, 3, 3.5, 4]
# generate the cells: explore different rect lengths:
aRect = [50, 60, 70, 80, 90, 100]
# generate the cells: explore different noise colors:
aNoise = [0, 25, 50, 100, 150, 200]


#generates grid given img
def getW(img):
    W = Device('wafer')
    px = 0
    py = 0

    for n in range(len(aDR)):
        # configure the cell parameters
        mysolar.dr_shift = aDR[n]
        mysolar.name = '6.2600 S23 -=-  #' +str(n)+'  -=-  DR_S='+str(aDR[n])   
        D = solarcell(mysolar, img)
        #D.write_gds('partyblob'+str(aDR[n])+'.gds')
        W << D.move( (px*mysolar.width,py*mysolar.height) )
        px=px+1
        if (px>4):
            px=0
            py=py+1
    
    #reset before next step
    mysolar.dr_shift = 0
    for n in range(len(aDR_range)):
        # configure the cell parameters
        mysolar.dr_expand = aDR_range[n]
        mysolar.name = '6.2600 S23 -=-  #' +str(n)+'  -=-  DR_R='+str(aDR_range[n])   
        D = solarcell(mysolar, img)
        #D.write_gds('partyblob'+str(aDR[n])+'.gds')
        W << D.move( (px*mysolar.width,py*mysolar.height) )
        px=px+1
        if (px>4):
            px=0
            py=py+1

    #reset before next step
    mysolar.dr_expand = 2
    mysolar.dr_shift = 0
    mysolar.noise_no = False
    for n in range(len(aNoise)):
        # configure the cell parameters
        mysolar.noise_color = aNoise[n]
        mysolar.name = '6.2600 S23 -=-  #' +str(n)+'  -=-  Noise='+str(aNoise[n])   
        D = solarcell(mysolar, img)
        #D.write_gds('partyblob'+str(aDR[n])+'.gds')
        W << D.move( (px*mysolar.width,py*mysolar.height) )
        px=px+1
        if (px>4):
            px=0
            py=py+1

    #reset before next step
    mysolar.dr_shift = 0
    mysolar.dr_expand = 2
    mysolar.rect_no = False
    mysolar.noise_no = True
    for n in range(len(aRect)):
        # configure the cell parameters
        mysolar.rect_size = aRect[n]
        mysolar.name = '6.2600 S23 -=-  #' +str(n)+'  -=-  Rect='+str(aRect[n])   
        D = solarcell(mysolar, img)
        #D.write_gds('partyblob'+str(aRect[n])+'.gds')
        W << D.move( (px*mysolar.width,py*mysolar.height) )
        px=px+1
        if (px>4):
            px=0
            py=py+1

    #get benchmark cells
    for i in [13]:
        D = solarcellS23.getCellN(i)
        W << D.move( (px*mysolar.width,py*mysolar.height) )
        px=px+1
        if (px>4):
            px=0
            py=py+1

    #get orig img
    """D = img2cell.get_orig_img(img)
    W << D.move( (px*mysolar.width,py*mysolar.height) )
    px=px+1
    if (px>4):
        px=0
        py=py+1
        """

    return W


################################################################
# generate the cells: explore different images:
getW('partyblob.jpg').write_gds('cellsgrid.gds')
#solarcell(mysolar, 'partyblob.jpg').write_gds('partyblob.gds')
#solarcell(mysolar, 'scrunge.jpg').write_gds('scrunge.gds')
#solarcell(mysolar, 'catsby-3.jpg').write_gds('catsby.gds')
#solarcell(mysolar, 'sevt.png').write_gds('sevt.gds')
#solarcell(mysolar, 'megamind.png').write_gds('megamind.gds')
