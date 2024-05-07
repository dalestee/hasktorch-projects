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

-- Define the trainingData list with batchSize tuples
trainingData :: Int ->[([Float], Float)]
trainingData batchSize = take batchSize $ cycle $ [([1.0],2.0),([2.0],4.0),([3.0],6.0),([1.0],2.0),([3.0],7.0), ([4.0],8.0)]


-- Define the function to process each tuple in the trainingData list
processTuple :: ([Float], Float) -> IO ()
processTuple (x, y) = do
    putStrLn $ "Input: " ++ show x ++ ", Output: " ++ show y

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 1 1   

-- Define the main function where the loop will be executed
main :: IO ()
main = do
    putStrLn "Looping through trainingData:"
    initModel <- sample $ createModel device -- extract LinearParams from IO action
    -- batchSize is the number of tuples in the trainingData list
    let batchLoss = foldLoop (trainingData batchSize) zeroTensor $ \(input, output) loss ->
                        let y' = linearLayer initModel $ asTensor'' device input
                            y = asTensor'' device output
                        in add loss $ mseLoss y y'
        lossValue = asValue batchLoss :: Float
    showLoss 1 1 lossValue
    where
    batchSize = 2
    learningRate = 1e-1
    optimizer = GD
    device = Device CPU 0