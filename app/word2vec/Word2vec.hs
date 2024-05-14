{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use newtype instead of data" #-}

module Word2vec (word2vec) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding')
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor ( Tensor, asTensor )
import Torch.TensorFactories (eye', zeros')

-- your text data (try small data first)
textFilePath :: String
textFilePath = "app/word2vec/data/review-texts.txt"
modelPath :: String
modelPath =  "app/word2vec/models/sample_model.pt"
wordLstPath :: String
wordLstPath = "app/word2vec/data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


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
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))

toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} =
  eye' wordNum wordDim


word2vec :: IO ()
word2vec = do
  -- load text file
  texts <- B.readFile textFilePath

  -- create word lst (unique)
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
  -- print wordlst

  -- create embedding(wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 2}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }

  putStrLn "Finish creating embedding"

  -- save params
  saveParams emb modelPath
  -- save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

  -- load params
  initWordEmb <- makeIndependent $ zeros' [1]
  let initEmb = Embedding {wordEmbedding = initWordEmb}
  loadedEmb <- loadParams initEmb modelPath

  let sampleTxt = B.pack $ encode "i love playing tapped out because it is fun to watch the town grow by earning"
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- convert to embedding
      embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)
  print embTxt