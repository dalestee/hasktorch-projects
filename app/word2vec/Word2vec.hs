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
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List ( nub, init, sortOn, sortBy )

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional( mseLoss, softmax, Dim(..), mseLoss, softmax, Dim(..) )
import Torch.NN( Parameterized(..), Parameter, flattenParameters, sample )
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor
    ( Tensor,
      asTensor,
      asValue,
      shape,
      dtype,
      device,
      shape,
      dtype,
      device )
import Torch.TensorFactories (eye', zeros')
import Debug.Trace (traceShow)

import ML.Exp.Chart (drawLearningCurve)

--hasktorch
import Torch.Optim (foldLoop, mkAdam)
import Torch.Device(Device(..), DeviceType(..))
import Torch.Train (update)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)
import Torch.DType (DType(..), DType(..) )

import Control.Monad (when)
import Prelude hiding (init)
import Data.Vector.Fusion.Bundle (inplace)
import Data.ByteString (ByteString, length)
import Data.ByteString.Char8 (pack)
import Data.ByteString.Lazy (fromStrict)
import Data.Ord (comparing, Down (Down))
import Torch.Typed (TensorListFold(TensorListFold))

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

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
initDataSets wordLines wordlst = zip input output
  where
      dictLength = Prelude.length wordlst
      wordToIndex = wordToIndexFactory $ nub wordlst
      wordsByOccurence = take (dictLength `div` 20) $ sortByOccurence $ concat wordLines
      filteredWordlst = filter (`elem` wordsByOccurence) wordlst
      input = [oneHotEncode (wordToIndex word) dictLength | word <- init filteredWordlst]
      output = [oneHotEncode (wordToIndex word) dictLength | word <- tail filteredWordlst]

getTopWords :: B.ByteString -> (B.ByteString -> Int) -> MLPParams -> Dim -> Int -> [B.ByteString] -> [(B.ByteString, Float)]
getTopWords word wordToIndex loadedEmb dim numWords wordlst = top10Words
  where
    wordIndex = wordToIndex word
    input = oneHotEncode wordIndex numWords
    output = forward loadedEmb dim input
    wordOutputed = M.toList $ M.fromList $ zip wordlst (asValue output :: [Float])
    sortedWords = sortBy (comparing (Down . snd)) wordOutputed
    top10Words = take 10 sortedWords

word2vec :: IO ()
word2vec = do

  --------------------------------------------------------------------------------
  -- Data
  --------------------------------------------------------------------------------

  texts <- B.readFile textFilePath

  -- create word lst (unique)
  let wordLines = preprocess texts
  let wordlst = take numWords $ concat wordLines
  let wordToIndex = wordToIndexFactory $ nub wordlst
  let indexesOfwordlst = map wordToIndex wordlst

  -- print $ "Word list: " ++ show wordlst
  -- print $ "Indexes of word list: " ++ show indexesOfwordlst
  -- print wordLines
  putStrLn ("Training data size: " ++ show (Prelude.length wordlst))
  -- create embedding(mlp with wordDim × wordNum)

  putStrLn "Finish creating embedding"

  let trainingData = initDataSets wordLines wordlst

  -- print $ "Training data: " ++ show trainingData

  --------------------------------------------------------------------------------
  -- Training
  --------------------------------------------------------------------------------

  -- initEmb <- loadParams hyperParams "app/cifar/models/model-cifar-256x2563850.pt" -- comment if you want to train from scratch
  initEmb <- sample hyperParams -- comment if you want to load a model
  let opt = mkAdam itr beta1 beta2 (flattenParameters initEmb)

  putStrLn "Start training"
  (trained, _, losses) <- foldLoop (initEmb, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
    let epochLoss = sum (map (loss model dim) trainingData)
    when (i `mod` 1 == 0) $ do
        print ("Epoch: " ++ show i ++ " | Loss: " ++ show (asValue epochLoss :: Float))
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
  loadedEmb <- loadParams initEmb modelPath

  -- print $ "Params: " ++ show (flattenParameters loadedEmb)

  let word = B.fromStrict $ pack "mcaffee"
  let top10Words = getTopWords word wordToIndex loadedEmb dim numWords wordlst

  print $ "Word: " ++ show word
  print $ "Output: " ++ show top10Words

  -- accuracy on treining data

  

  where
    numEpochs = 300 :: Int
    numWords = 770 :: Int
    wordDim = 8 :: Int

    textFilePath :: String
    textFilePath = "app/word2vec/data/lorem.txt"
    modelPath :: String
    modelPath =  "app/word2vec/models/sample_model.pt"
    wordLstPath :: String
    wordLstPath = "app/word2vec/data/sample_wordlst.txt"

    device = Device CPU 0
    hyperParams = MLPHypParams device numWords [(wordDim, Relu), (numWords, Id)]

    -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
    itr = 0 :: Int
    beta1 = 0.9 :: Float
    beta2 = 0.999 :: Float
    lr = 1e-1 :: Tensor
    dim = Dim (-1) :: Dim