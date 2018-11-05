from utils import *
import tqdm
import time
from scipy.optimize import fmin_l_bfgs_b
from Settings import *
import keras.backend as K
import copy

def calculate_loss(outputImg):
    if outputImg.shape != (1, WIDTH, WIDTH, 3):
        outputImg = outputImg.reshape((1, WIDTH, HEIGHT, 3))
    loss_fcn = K.function([outModel.input], [get_total_loss(outModel.input)])
    return loss_fcn([outputImg])[0].astype('float64')

def get_total_loss(outputPlaceholder):
    F = get_feature_reps(outputPlaceholder,layer_names=[contentLayerNames], model=outModel)[0]
    Gs = get_feature_reps(outputPlaceholder,layer_names=styleLayerNames, model=outModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss

def get_content_loss(F, P):
    if lossType=='SE':
        cLoss = 0.5*K.sum(K.square(F - P))
    else:
        cLoss= 0.5*K.sum(K.abs(F - P))
    return cLoss

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        if lossType=='SE':
            sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
        else:
            sLoss += w * 0.25 * K.sum(K.abs(G_gram - A_gram)) / (N_l ** 2 * M_l ** 2)
    return sLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, WIDTH,HEIGHT, 3):
        gImArr = gImArr.reshape((1, WIDTH,HEIGHT, 3))
    grad_fcn = K.function([outModel.input], K.gradients(get_total_loss(outModel.input), [outModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad

def get_feature_reps(x,layer_names, model):

    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=K.get_session())
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def callbackF(Xi):
    """A call back function for scipy optimization to record Xi each step"""
    global iterator
    global count

    if record:
        deepCopy=copy.deepcopy(Xi)
        this_styleLoss = calculate_style_loss(deepCopy)
        this_contentLoss = calculate_content_loss(deepCopy)
        contentLoss.append(this_contentLoss)
        styleLoss.append(this_styleLoss)
        this_totalLoss=this_styleLoss*beta+this_contentLoss*alpha
        totalLossList.append(this_totalLoss)

    if iterator%50==0:
        deepCopy=copy.deepcopy(Xi)
        i = iterator // 50
        xOut = postprocess_array(deepCopy)
        imgName = PATH_OUTPUT + '.'.join(name_list[:-1]) + '_{}.{}'.format(
            str(i) if i!=stop-1 else 'final', name_list[-1])
        _ = save_original_size(xOut, imgName, contentOrignialImgSize)

    iterator+=1
    count.update(1)

if __name__=='__main__':

    parser=build_parser()
    args=parser.parse_args()
    content_name=args.content
    style_name=args.style
    output_name=args.output
    iteration=int(args.iter)
    flw=int(args.rstep)
    lossType=args.losstype
    record=False if args.record=='F' else True
    rstep=args.rstep
    stop = iteration // 50
    alpha=args.alpha
    beta=args.beta

    contentLossList=[]
    styleLossList=[]
    totalLossList=[]




    contentImgArr, contentOrignialImgSize = inputImageUtils(PATH_INPUT_CONTENT+content_name, SIZE)
    styleImgArr, styleOrignialImgSize = inputImageUtils(PATH_INPUT_STYLE+style_name, SIZE)
    output, outputPlaceholder = outImageUtils(WIDTH, HEIGHT)
    contentModel, styleModel, outModel = BuildModel(contentImgArr, styleImgArr, outputPlaceholder)

    P = get_feature_reps(x=contentImgArr,layer_names=[contentLayerNames], model=contentModel)[0]
    As = get_feature_reps(x=styleImgArr,layer_names=styleLayerNames, model=styleModel)
    ws = wlList[flw]


    outputImg=output.flatten()
    start=time.time
    count=tqdm.tqdm(total=iteration)
    name_list = output_name.split('.')
    iterator=1
    xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, outputImg, fprime=get_grad,
                                maxiter=iteration,disp=True,callback=callbackF)
    if record:
        plt.plot(totalLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('TotalLoss')
        plt.savefig(PATH_OUTPUT + 'TotalLoss.jpg')

        plt.figure()
        plt.plot(contentLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('ContentLoss')
        plt.savefig(PATH_OUTPUT + 'ContentLoss.jpg')

        plt.figure()
        plt.plot(styleLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('StyleLoss')
        plt.savefig(PATH_OUTPUT + 'StyleLoss.jpg')
