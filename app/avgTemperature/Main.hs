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

parseData :: FilePath -> [(Float)]
parseData filePath = map parseLine $ tail $ lines $ unsafePerformIO $ readFile filePath
  where
    parseLine :: String -> Float
    parseLine line = read $ last $ splitOn "," line

trainingData :: [(Float)]
trainingData = parseData "data/train.csv"

testData :: [(Float)]
testData = parseData "data/valid.csv"

main :: IO()
main = do
    -- parse training data and print the first 5
    -- print $ take 5 (parseData "data/valid.csv")
    -- print $ take 5 (parseData "data/train.csv")
    let iter = 100::Int
        device = Device CPU 0
    initModel <- sample $ LinearHypParams device True 1 1
    ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
        let batchLoss = foldLoop trainingData zeroTensor $ \input loss ->
                            let y' = linearLayer model $ asTensor'' device [input]
                                y = asTensor'' device [input]
                            in add loss $ mseLoss y y'
            lossValue = (asValue batchLoss)::Float
        showLoss 1 epoc lossValue
        u <- update model opt batchLoss 10e-7
        return (u, lossValue)
    saveParams trainedModel "regression.model"

    drawLearningCurve "graph-reg.png" "Learning Curve" [("",reverse losses)]

    loadedModel <- loadParams (LinearHypParams device True 1 1) "regression.model"
    print loadedModel