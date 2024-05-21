{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# HLINT ignore "Fuse mapM/map" #-}
{-# HLINT ignore "Missing NOINLINE pragma" #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# HLINT ignore "Use second" #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

module TextTreatment (normalizeText, removeParticuleWords) where

import Data.ByteString as B
import Data.Text as T
import Data.Text.Encoding as E
import Data.Char (isAlpha, isSpace)

normalizeText :: FilePath -> IO ()
normalizeText filePath = do
  content <- B.readFile filePath
  let normalizedContent = normalize (E.decodeUtf8 content)
  B.writeFile filePath (E.encodeUtf8 normalizedContent)
  where
    normalize text = T.toLower (T.filter (\c -> isAlpha c || isSpace c) text)

removeParticuleWords :: FilePath -> IO ()
removeParticuleWords filePath = do
  content <- B.readFile filePath
  let normalizedContent = removeParticules (E.decodeUtf8 content)
  B.writeFile filePath (E.encodeUtf8 normalizedContent)
  where
    removeParticules text = T.unwords (Prelude.filter (\word -> not (word `Prelude.elem` (Prelude.map T.pack particules))) (T.words text))    
    particules = ["this", "it", "that", "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "being", "have", "has", "had", "do", "does", "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could", "of", "in", "on", "at", "to", "for", "with", "about", "by", "from", "as"]

main :: IO ()
main  = do
  normalizeText "app/word2vec/data/review-texts-original.txt"
  -- removeParticuleWords "app/word2vec/data/review-texts.txt"