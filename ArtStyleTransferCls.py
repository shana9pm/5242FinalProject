from utils import *
import tqdm
from scipy.optimize import fmin_l_bfgs_b
from Settings import *
import keras.backend as K
import copy
import matplotlib.pyplot as plt
import json

#Note that get/calculate styleLoss/contentLoss only useful in record=True mode.
#Normaly we would only run to get full loss so if we want to get part loss we need to feedfoward again

class Styletransfer:

    def __init__(self,args):

        #paramter from parser
        self.content_name = args.content
        self.style_name = args.style
        self.output_name = args.output
        self.iteration = int(args.iter)
        flw=int(args.flw)
        self.ws = wlList[flw]
        self.styleLossType=args.styleloss
        self.contentLossType=args.contentloss
        self.record=False if args.record=="F" else True
        self.rstep=int(args.rstep)

        #get input pictures and get features
        contentImgArr, contentOrignialImgSize = inputImageUtils(PATH_INPUT_CONTENT + self.content_name, SIZE)
        styleImgArr, styleOrignialImgSize = inputImageUtils(PATH_INPUT_STYLE + self.style_name, SIZE)
        output, outputPlaceholder = outImageUtils(WIDTH, HEIGHT)
        self.contentOriginalImgSize=contentOrignialImgSize
        self.contentModel, self.styleModel, self.outModel = BuildModel(contentImgArr, styleImgArr, outputPlaceholder)
        self.outputImg = output.flatten()
        #P and As are constant tensors,not placeholder
        self.P = self.get_feature_reps(x=contentImgArr, layer_names=[contentLayerNames], model=self.contentModel)[0]
        self.As = self.get_feature_reps(x=styleImgArr, layer_names=styleLayerNames, model=self.styleModel)

        #some paramters for operation
        self.count = tqdm.tqdm(total=self.iteration)
        self.name_list = self.output_name.split('.')
        self.contentLoss=[]
        self.styleLoss=[]
        self.totalLoss=[]
        self.totalLoss2=[]
        self.stop = self.iteration // self.rstep

    def main(self):
        #xopt, f_val, info = fmin_l_bfgs_b(self.calculate_loss, self.outputImg, fprime=self.get_grad,
        #                                  maxiter=self.iteration, disp=True, callback=self.callbackF)
        for i in range(self.iteration):
            self.outputImg, f_val, info=fmin_l_bfgs_b(self.calculate_loss, self.outputImg, fprime=self.get_grad,maxiter=1,
                                          disp=False)
            if self.record:
                deepCopy=copy.deepcopy(self.outputImg)
                this_styleLoss = self.calculate_style_loss(deepCopy)
                this_contentLoss = self.calculate_content_loss(deepCopy)
                this_totalLoss=self.calculate_loss(deepCopy)
                self.contentLoss.append(this_contentLoss)
                self.styleLoss.append(this_styleLoss)
                self.totalLoss.append(this_totalLoss)
                self.totalLoss2.append(float(f_val))

            if i % self.rstep == 0:
                deepCopy = copy.deepcopy(self.outputImg)
                iter = i// self.rstep
                xOut = postprocess_array(deepCopy)
                imgName = PATH_OUTPUT + '.'.join(self.name_list[:-1]) + '_{}.{}'.format(
                    str(iter) if iter != self.stop else 'final', self.name_list[-1])
                _ = save_original_size(xOut, imgName, self.contentOriginalImgSize)
            self.count.update(1)

        if self.record:
            plt.plot(self.totalLoss)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('TotalLoss')
            plt.savefig(PATH_OUTPUT+'TotalLoss.jpg')

            plt.figure()
            plt.plot(self.contentLoss)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('ContentLoss')
            plt.savefig(PATH_OUTPUT+'ContentLoss.jpg')

            plt.figure()
            plt.plot(self.styleLoss)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('StyleLoss')
            plt.savefig(PATH_OUTPUT+'StyleLoss.jpg')
            print(self.contentLoss)
            print(self.styleLoss)
            print(self.totalLoss)
            print(self.totalLoss2)
        self.recordPara()



    def calculate_loss(self,outputImg):
        if outputImg.shape != (1, WIDTH, WIDTH, 3):
            outputImg = outputImg.reshape((1, WIDTH, HEIGHT, 3))
        loss_fcn = K.function([self.outModel.input], [self.get_total_loss(self.outModel.input)])
        result = loss_fcn([outputImg])[0].astype('float64')

        return result

    def get_total_loss(self,outputPlaceholder, alpha=1.0, beta=10000.0):
        F = self.get_feature_reps(outputPlaceholder, layer_names=[contentLayerNames], model=self.outModel)[0]
        Gs = self.get_feature_reps(outputPlaceholder, layer_names=styleLayerNames, model=self.outModel)
        contentLoss = self.get_content_loss(F)
        styleLoss = self.get_style_loss(Gs)
        totalLoss = alpha * contentLoss + beta * styleLoss
        return totalLoss

    """Note that we only support Square error and Absolute error"""

    def get_content_loss(self,F):
        if self.contentLossType=='SE':
            cLoss = 0.5 * K.sum(K.square(F - self.P))
        else:
            cLoss = 0.5 * K.sum(K.abs(F - self.P))
        return cLoss

    def get_style_loss(self,Gs):
        sLoss = K.variable(0.)
        if self.styleLossType=='SE':
            for w, G, A in zip(self.ws, Gs, self.As):
                M_l = K.int_shape(G)[1]
                N_l = K.int_shape(G)[0]
                G_gram = self.get_Gram_matrix(G)
                A_gram = self.get_Gram_matrix(A)
                sLoss += w * 0.25 * K.sum(K.square(G_gram - A_gram)) / (N_l ** 2 * M_l ** 2)
        else:
            for w, G, A in zip(self.ws, Gs, self.As):
                M_l = K.int_shape(G)[1]
                N_l = K.int_shape(G)[0]
                G_gram = self.get_Gram_matrix(G)
                A_gram = self.get_Gram_matrix(A)
                sLoss += w * 0.25 * K.sum(K.abs(G_gram - A_gram)) / (N_l ** 2 * M_l ** 2)
        return sLoss

    def get_Gram_matrix(self,F):
        G = K.dot(F, K.transpose(F))
        return G

    def get_grad(self,gImArr):
        """
        Calculate the gradient of the loss function with respect to the generated image
        """
        if gImArr.shape != (1, WIDTH, HEIGHT, 3):
            gImArr = gImArr.reshape((1, WIDTH, HEIGHT, 3))
        grad_fcn = K.function([self.outModel.input], K.gradients(self.get_total_loss(self.outModel.input), [self.outModel.input]))
        grad = grad_fcn([gImArr])[0].flatten().astype('float64')
        return grad

    def get_feature_reps(self,x, layer_names, model):

        featMatrices = []
        for ln in layer_names:
            selectedLayer = model.get_layer(ln)
            featRaw = selectedLayer.output
            featRawShape = K.shape(featRaw).eval(session=K.get_session())
            N_l = featRawShape[-1]
            M_l = featRawShape[1] * featRawShape[2]
            featMatrix = K.reshape(featRaw, (M_l, N_l))
            featMatrix = K.transpose(featMatrix)
            featMatrices.append(featMatrix)
        return featMatrices
    '''
    def callbackF(self,Xi):
        """A call back function for scipy optimization to record Xi each step"""

        if self.record:
            deepCopy = copy.deepcopy(Xi)
            this_styleLoss = self.calculate_style_loss(deepCopy)
            this_contentLoss = self.calculate_content_loss(deepCopy)
            this_totalLoss=self.calculate_loss(deepCopy)
            self.contentLoss.append(this_contentLoss)
            self.styleLoss.append(this_styleLoss)
            self.totalLoss.append(this_totalLoss)
        if self.iterator % self.rstep == 0:
            deepCopy = copy.deepcopy(Xi)
            i = int(self.iterator / self.rstep)
            xOut = postprocess_array(deepCopy)
            imgName = PATH_OUTPUT + '.'.join(self.name_list[:-1]) + '_{}.{}'.format(
                str(i) if i != self.stop else 'final', self.name_list[-1])
            _ = save_original_size(xOut, imgName, self.contentOriginalImgSize)

        self.iterator += 1
        self.count.update(1)
    '''

    """The following functions are used for calculation of loss"""
    def get_style_loss_forward(self,outputPlaceholder):
        Gs = self.get_feature_reps(outputPlaceholder, layer_names=styleLayerNames, model=self.outModel)
        styleLoss = self.get_style_loss(Gs)
        return styleLoss

    def get_content_loss_forward(self,outputPlaceholder):
        F = self.get_feature_reps(outputPlaceholder, layer_names=[contentLayerNames], model=self.outModel)[0]
        contentLoss = self.get_content_loss(F)
        return contentLoss

    def calculate_style_loss(self,Xi):
        if Xi.shape != (1, WIDTH, WIDTH, 3):
            Xi = Xi.reshape((1, WIDTH, HEIGHT, 3))
        loss_fcn = K.function([self.outModel.input], [self.get_style_loss_forward(self.outModel.input)])
        return loss_fcn([Xi])[0].astype('float64')

    def calculate_content_loss(self,Xi):
        if Xi.shape != (1, WIDTH, WIDTH, 3):
            Xi = Xi.reshape((1, WIDTH, HEIGHT, 3))
        loss_fcn = K.function([self.outModel.input], [self.get_content_loss_forward(self.outModel.input)])
        return loss_fcn([Xi])[0].astype('float64')

    def recordPara(self):
        paraDict={'contentName':self.content_name,
                  'styleName':self.style_name,
                  'outputName':self.output_name,
                  'maxIter':self.iteration,
                  'flw':self.ws,
                  'styleLossType':self.styleLossType,
                  'contentLossType':self.contentLossType,
                  'rstep':self.rstep
                  }
        with open('parameters.txt','w') as f:
            f.write(json.dumps(paraDict))




parser = build_parser()
args = parser.parse_args()
cls=Styletransfer(args)
cls.main()