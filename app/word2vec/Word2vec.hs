{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use newtype instead of data" #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# OPTIONS_GHC -Wno-unused-matches #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}
{-# LANGUAGE DeriveGeneric #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}

module Word2vec (word2vec) where
import Codec.Binary.UTF8.String                ( encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word                               ( Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List                               ( nub, init, sortOn, sortBy, elemIndex )

import Torch.Autograd                          ( makeIndependent, toDependent )
import Torch.Functional                        ( mseLoss, softmax, Dim(..), mseLoss, softmax, Dim(..) )
import Torch.NN                                ( Parameterized(..), Parameter, flattenParameters, sample, Linear(..) )
import Torch.Serialize                         ( saveParams )
import Torch.Train                             ( loadParams, update )
import Torch.Tensor                            ( Tensor, asTensor, asValue, shape, dtype, device, shape, dtype, device )
import Torch.TensorFactories                   ( eye', zeros' )
import Debug.Trace                             ( traceShow )

import ML.Exp.Chart                            ( drawLearningCurve )

import MatrixOp (magnitude, dotProduct)

--hasktorch
import Torch.Optim                             ( foldLoop, mkAdam )
import Torch.Device                            ( Device(..), DeviceType(..) )
import Torch.Layer.MLP                         ( MLPHypParamsNoBias(..), MLPHypParams(..), ActName(..), mlpLayer, MLPParams(..) )
import Torch.DType                             ( DType(..), DType(..) )

import Control.Monad                           ( when )
import Prelude hiding                          ( init )
import Data.Vector.Fusion.Bundle               ( inplace )
import Data.ByteString                         ( ByteString, length )
import Data.ByteString.Char8                   ( pack )
import Data.ByteString.Lazy                    ( fromStrict )
import Data.Ord                                ( comparing, Down (Down) )
import Torch.Typed                             ( TensorListFold(TensorListFold) )
import Torch.Layer.Linear                      ( LinearParams(LinearParams) )


isUnncessaryChar ::
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` map (head . encode) [".", "!"]

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (Prelude.length wordlst) wrd (M.fromList (zip wordlst [0..]))

loss :: MLPParams -> Dim -> (Tensor, Tensor) -> Tensor
loss model dim (input, output) =
    let y = forward model dim input
    in mseLoss y output

forward :: MLPParams -> Dim -> Tensor -> Tensor
forward emb dim input = softmax dim output
    where
        output = mlpLayer emb input

sortByOccurence :: [B.ByteString] -> [B.ByteString]
sortByOccurence wordlst = reverse $ map fst $ sortOn snd $ M.toList $ M.fromListWith (+) [(w, 1) | w <- wordlst]

setAt :: Int -> a -> [a] -> [a]
setAt idx val lst = take idx lst ++ [val] ++ drop (idx + 1) lst

oneHotEncode :: Int -> Int -> Tensor
oneHotEncode index size = asTensor $ setAt index 1 (zeros :: [Float])
  where
    zeros = replicate size 0

initDataSets :: [[B.ByteString]] -> [B.ByteString] -> [(Tensor, Tensor)]
initDataSets wordLines wordlst = pairs
  where
      dictLength = Prelude.length wordlst
      wordToIndex = wordToIndexFactory $ nub wordlst
      input = concatMap createInputPairs wordlst
      output = concatMap createOutputPairs wordlst
      pairs = zip input output
      createInputPairs word = case getIndex wordlst word of
        Just idx -> replicate 4 (oneHotEncode (wordToIndex word) dictLength)
        Nothing -> []
      createOutputPairs word = case getIndex wordlst word of
        Just idx -> map (\i -> if i >= 0 && i < Prelude.length wordlst then oneHotEncode (wordToIndex (wordlst !! i)) dictLength else zeros' [dictLength]) [idx - 1, idx + 1, idx - 2, idx + 2]
        Nothing -> []
      getIndex lst word = elemIndex word lst

-- initDataSets :: [[B.ByteString]] -> [B.ByteString] -> [(Tensor, Tensor)]
-- initDataSets wordLines wordlst = pairs
--   where
--       dictLength = Prelude.length wordlst
--       wordToIndex = wordToIndexFactory $ nub wordlst
--       input = concatMap createInputPairs wordlst
--       output = concatMap createOutputPairs wordlst
--       pairs = zip input output
--       createInputPairs word = case getIndex wordlst word of
--         Just idx -> replicate 4 (oneHotEncode (wordToIndex word) dictLength)
--         Nothing -> []
--       createOutputPairs word = case getIndex wordlst word of
--         Just idx -> if idx + 1 >= 0 && idx + 1 < Prelude.length wordlst 
--                     then [oneHotEncode (wordToIndex (wordlst !! (idx + 1))) dictLength] 
--                     else []
--         Nothing -> []
--       getIndex lst word = elemIndex word lst

getTopWords :: B.ByteString -> (B.ByteString -> Int) -> MLPParams -> Dim -> Int -> [B.ByteString] -> [(B.ByteString, Float)]
getTopWords word wordToIndex loadedEmb dim numWords wordlst = top10Words
  where
    wordIndex = wordToIndex word
    input = oneHotEncode wordIndex numWords
    output = forward loadedEmb dim input
    wordOutputed = M.toList $ M.fromList $ zip wordlst (asValue output :: [Float])
    sortedWords = sortBy (comparing (Down . snd)) wordOutputed
    top10Words = take 10 sortedWords

printOutputLayerWeights :: MLPParams -> IO ()
printOutputLayerWeights (MLPParams layers) =
  case reverse layers of
    [] -> putStrLn "No layers found!"
    ((LinearParams weight _, _):_) -> do
      let weights = toDependent weight
      putStrLn "Weights of the output layer:"
      print weights

getOutputLayerWeights :: Parameterized p => p -> Tensor
getOutputLayerWeights model =
  let linearParams = flattenParameters model
  in case reverse linearParams of
    [] -> asTensor ([] :: [Float])
    (param:_) -> toDependent param

-- similarity between two wordsVec
simCosin :: [Float] -> [Float] -> Float
simCosin vec1 vec2 = dotProduct vec1 vec2 / (magnitude vec1 * magnitude vec2)

lookForMostSimilarWords :: [Float] -> [[Float]] -> [B.ByteString] -> [(B.ByteString, Float)]
lookForMostSimilarWords wordVec wordVecs wordlst = top10Words
  where
    wordSim = map (simCosin wordVec) wordVecs
    wordSimWithIndex = zip wordSim [0..]
    sortedWords = sortBy (comparing (Down . fst)) wordSimWithIndex
    top10Words = take 10 $ map (\(sim, idx) -> (wordlst !! idx, sim)) sortedWords

word2vec :: IO ()
word2vec = do

  --------------------------------------------------------------------------------
  -- Data
  --------------------------------------------------------------------------------

  texts <- B.readFile textFilePath

  -- -- create word lst (unique)
  let wordLines = preprocess texts
  let wordlst = take numWords $ concat wordLines
  let wordToIndex = wordToIndexFactory $ nub wordlst
  -- let indexesOfwordlst = map wordToIndex wordlst

  -- putStrLn "Finish creating embedding"

  let trainingData = initDataSets wordLines wordlst

  print $ "Training data size: " ++ show (Prelude.length trainingData)

  --------------------------------------------------------------------------------
  -- Training
  --------------------------------------------------------------------------------

  -- initEmb <- loadParams hyperParamsNoBias "app/word2vec/models/sample_model.pt" -- comment if you want to train from scratch
  initEmb <- sample hyperParamsNoBias -- comment if you want to load a model
  -- initEmb' <- sample hyperParams
  -- print $ "ParamsNoBias: " ++ show (flattenParameters initEmb)
  -- putStrLn "\n"
  -- print $ "ParamsBias: " ++ show (flattenParameters initEmb')
  let opt = mkAdam itr beta1 beta2 (flattenParameters initEmb)

  putStrLn "Start training"
  (trained, _, losses) <- foldLoop (initEmb, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
    let epochLoss = sum (map (loss model dim) trainingData)
    when (i `mod` 1 == 0) $ do
        print ("Epoch: " ++ show i ++ " | Loss: " ++ show (asValue epochLoss :: Float))
    -- when (i `mod` 50 == 0) $ do
    --     saveParams model (modelPath ++ "-" ++ show i ++ ".pt")
    (newState, newOpt) <- update model optimizer epochLoss lr
    return (newState, newOpt, losses :: [Float]) -- without the losses curve
    -- return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

--------------------------------------------------------------------------------
  -- Saving
  --------------------------------------------------------------------------------

  -- save params
  saveParams trained modelPath
  -- save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

  --------------------------------------------------------------------------------
  -- Testing
  --------------------------------------------------------------------------------

  -- load params
  loadedEmb <- loadParams hyperParamsNoBias modelPath

  let vectorDic :: [[Float]]
      vectorDic = asValue $ getOutputLayerWeights loadedEmb

  let word1 = B.fromStrict $ pack "apps"
  let top10Words1 = getTopWords word1 wordToIndex loadedEmb dim numWords wordlst

  let wordIndex = wordToIndex word1
  let wordVec = vectorDic !! wordIndex
  let mostSimilarWords = lookForMostSimilarWords wordVec vectorDic wordlst
  
  print $ "Word: " ++ show word1
  print $ "Most similar words to " ++ show word1 ++ ": " ++ show mostSimilarWords
  -- print $ "Output: " ++ show top10Words1

  print "Finish"

  -- accuracy on treining data

  -- let accuracy = sum [1 | (input, output) <- trainingData, let y = forward loadedEmb dim input, let y' = asValue y :: [Float], let output' = asValue output :: [Float], y' == output'] / fromIntegral (Prelude.length trainingData)

  -- print $ "Accuracy: " ++ show accuracy

  where
    numEpochs = 200 :: Int
    numWords = 15000 :: Int
    wordDim = 16 :: Int

    textFilePath :: String
    textFilePath = "app/word2vec/data/review-texts.txt"
    modelPath :: String
    modelPath =  "app/word2vec/models/sample_model.pt"
    wordLstPath :: String
    wordLstPath = "app/word2vec/data/sample_wordlst.txt"

    device = Device CPU 0
    hyperParams = MLPHypParams device numWords [(wordDim, Relu), (numWords, Id)]
    hyperParamsNoBias = MLPHypParamsNoBias hyperParams

    -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
    itr = 0 :: Int
    beta1 = 0.9 :: Float
    beta2 = 0.999 :: Float
    lr = 1e-1 :: Tensor
    dim = Dim 0 :: Dim