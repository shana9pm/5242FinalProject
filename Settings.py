
PATH_INPUT_STYLE = 'input/style/'
PATH_INPUT_CONTENT = 'input/content/'
PATH_OUTPUT = 'output/'
WIDTH=512
HEIGHT=512
SIZE=(512,512)

contentLayerNames = 'block4_conv2'

styleLayerNames = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
]

wlList=[
    [0.25, 0.25, 0.25, 0.25],#baseline
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [.5,.5,0, 0],
    [.5,0,.5, 0],
    [.5, 0,0,.5],
    [0,.5,.5,0],
    [0,.5,0,.5],
    [0,0,.5,.5],
    [1/3,1/3,1/3,0],
    [1/3,1/3,0,1/3],
    [1/3,0,1/3,1/3],
    [0,1/3,1/3,1/3],
]