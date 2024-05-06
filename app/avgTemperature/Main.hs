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
import Control.Monad (liftM)
import System.IO.Unsafe (unsafePerformIO)
import Data.List.Split (splitOn)


-- temperature (float) out of "data/train.csv"
-- date,daily_mean_temperature
-- 2023/4/1,16.0

-- takes 7 last days and predict the temperature of the next day
parseData :: FilePath -> [([Float],Float)]
parseData path = unsafePerformIO $ do
    content <- readFile path
    let temps = map (read . last . splitOn ",") $ tail $ lines content
    return [(take 7 (drop i temps), temps !! (i+7)) | i <- [0..(length temps - 8)]]

-- temperature of the 7 last days to predict the temperature of the next day
trainingData :: [([Float],Float)]
trainingData = parseData "data/train.csv"

testData :: [([Float],Float)]
testData = parseData "data/valid.csv"

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 7 1

main :: IO()
main = do
    -- parse training data and print the first 5
    -- print $ take 5 (parseData "data/valid.csv")
    -- print $ take 5 (parseData "data/train.csv")
    model <- initModel
    ((trainedModel,_),losses) <- mapAccumM [1..numIters] (model,optimizer) $ \epoc (model,opt) -> do
        let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                            let y' = linearLayer model $ asTensor'' device input
                                y = asTensor'' device output
                            in add loss $ mseLoss y y'
            lossValue = (asValue batchLoss)::Float
        showLoss 1 epoc lossValue
        u <- update model opt batchLoss 9e-11  
        let (updatedModel, updatedOptimizer) = u
        return ((updatedModel, updatedOptimizer), lossValue)
    saveParams trainedModel "regression.model"
    drawLearningCurve "graph-reg.png" "Learning Curve" [("",reverse losses)]


    where
    -- batchSize = 1
    optimizer = GD
    numIters = 200
    -- numFeatures = 3
    device = Device CPU 0
    initModel = sample $ createModel device
    