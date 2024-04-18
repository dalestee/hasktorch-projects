module Main where

--hasktorch
import Torch.Tensor (asValue)
import Torch.Functional (mseLoss,add)
import Torch.Device     (Device(..),DeviceType(..))
import Torch.NN         (sample)
import Torch.Optim      (GD(..))
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Train      (update,showLoss,zeroTensor,saveParams,loadParams)
import Torch.Control    (mapAccumM,foldLoop)
import Torch.Layer.Linear (LinearHypParams(..),linearLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools
import Control.Monad (when,liftM)

trainingData :: [([Float],Float)]
trainingData = [([1],2),([2],4),([3],6),([1],2),([3],7)]

testData :: [([Float],Float)]
testData = [([3],7)]

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 1 1

main :: IO()
main = do
    model <- initModel
    ((trainedModel,_),losses) <- mapAccumM [1..numIters] (model,optimizer) $ \epoc (model,opt) -> do
        let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                            let y' = linearLayer model $ asTensor'' device input
                                y = asTensor'' device output
                            in add loss $ mseLoss y y'
            lossValue = (asValue batchLoss)::Float
        showLoss 5 epoc lossValue
        u <- update model opt batchLoss 5e-4     
        let (updatedModel, updatedOptimizer) = u
        return ((updatedModel, updatedOptimizer), lossValue)
    saveParams trainedModel "regression.model"
    drawLearningCurve "graph-reg.png" "Learning Curve" [("",reverse losses)]


    where
    batchSize = 4
    optimizer = GD
    numIters = 200
    numFeatures = 3
    device = Device CPU 0
    initModel = sample $ createModel device
    
