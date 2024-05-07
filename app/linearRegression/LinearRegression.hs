module LinearRegression (linear) where

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
testData = [([x],2*x) | x <- [4..10]]

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 1 1

-- ...

linear :: IO()
linear = do
    -- model <- initModel
    model <- loadParams (LinearHypParams device True 1 1) "app/linearRegression/models/model-linear-good.pt"
    ((trainedModel,_),losses) <- mapAccumM [1..numIters] (model,optimizer) $ \epoc (model,opt) -> do
        let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                            let y' = linearLayer model $ asTensor'' device input
                                y = asTensor'' device output
                            in add loss $ mseLoss y y'
            lossValue = (asValue batchLoss)::Float
        showLoss 5 epoc lossValue
        u <- update model opt batchLoss learningRate
        let (updatedModel, updatedOptimizer) = u
        return ((updatedModel, updatedOptimizer), lossValue)
    let filename = "app/linearRegression/curves/graph-linear-mse" ++ show (last (reverse losses)) ++ ".png"
    drawLearningCurve filename "Learning Curve" [("", reverse losses)]
    let modelname = "app/linearRegression/models/model-linear-" ++ show (last (reverse losses)) ++ ".pt"
    saveParams trainedModel modelname

    --test

    model' <- loadParams (LinearHypParams device True 1 1) "app/linearRegression/models/model-linear-good.pt"
    -- model' <- loadParams (LinearHypParams device True 1 1) modelname
    let testAccuracy = foldLoop testData 0 $ \(input,output) acc ->
                        let y' = linearLayer model' $ asTensor'' device input
                            y = asTensor'' device output
                            diff = abs (asValue y - asValue y') :: Float
                        in if diff < 0.5 then acc + 1 else acc
    print $ "Test Accuracy: " ++ show (testAccuracy / fromIntegral (length testData))

    let trainAccuracy = foldLoop trainingData 0 $ \(input,output) acc ->
                        let y' = linearLayer model' $ asTensor'' device input
                            y = asTensor'' device output
                            diff = abs (asValue y - asValue y') :: Float
                        in if diff < 0.5 then acc + 1 else acc
    print $ "Train Accuracy: " ++ show (trainAccuracy / fromIntegral (length trainingData))

    where
    learningRate = 1e-2
    optimizer = GD
    numIters = 25
    device = Device CPU 0
    initModel = sample $ createModel device