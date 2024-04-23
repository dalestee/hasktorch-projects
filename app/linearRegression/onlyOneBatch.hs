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

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 1 1

main :: IO()
main = do 
    model <- initModel
    print model
    -- only one batch
    -- create input and output tensors
    let input = [1] :: [Float]
        output = 2 :: Float
    -- y' = model(x)
        y' = linearLayer model $ asTensor'' device input
    -- y = expected output
        y = asTensor'' device output
    -- calculate loss
        loss = mseLoss y y'
    putStrLn $ "batchLoss:"
    print loss
    putStrLn $ "----------------------------------------------"
    -- update model
    u <- update model optimizer loss 5e-4
    let (updatedModel, updatedOptimizer) = u
    print updatedModel

    where
    learningRate = 1e-1
    optimizer = GD
    numFeatures = 3
    device = Device CPU 0
    initModel = sample $ createModel device