{-# OPTIONS_GHC -Wno-unused-top-binds #-}
module Temperature (temperature) where

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
import Data.List.Split (splitOn)


-- temperature (float) out of "data/train.csv"
-- date,daily_mean_temperature
-- 2023/4/1,16.0

-- takes 7 last days and predict the temperature of the next day
parseData :: String -> [([Float],Float)]
parseData content =
    let temps = map (read . last . splitOn ",") $ tail $ lines content
    in [(take 7 (drop i temps), temps !! (i+7)) | i <- [0..(length temps - 8)]]

readData :: FilePath -> IO [([Float],Float)]
readData path = do
    content <- readFile path
    return $ parseData content

-- temperature of the 7 last days to predict the temperature of the next day
trainingData :: IO [([Float],Float)]
trainingData = readData "app/temperature/data/train.csv"

testData :: IO [([Float],Float)]
testData = readData "app/temperature/data/valid.csv"

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 7 1

temperature :: IO()
temperature = do
    
    print $ take 5 (parseData "app/temperature/data/valid.csv")
    print $ take 5 (parseData "app/temperature/data/train.csv")
    model <- initModel
    trainingDataList <- trainingData
    ((trainedModel,_),losses) <- mapAccumM [1..numIters] (model,optimizer) $ \epoc (model,opt) -> do
        let batchLoss = foldLoop trainingDataList zeroTensor $ \(input,output) loss ->
                            let y' = linearLayer model $ asTensor'' device input
                                y = asTensor'' device output
                            in add loss $ mseLoss y y'
            lossValue = asValue batchLoss::Float
        showLoss 1 epoc lossValue
        u <- update model opt batchLoss 1e-9
        let (updatedModel, updatedOptimizer) = u
        return ((updatedModel, updatedOptimizer), lossValue)
    saveParams trainedModel "app/temperature/models/temp-model.pt"
    drawLearningCurve "app/temperature/curves/graph-avg.png" "Learning Curve" [("",reverse losses)]

    where
    optimizer = GD
    numIters = 300
    device = Device CPU 0
    initModel = sample $ createModel device