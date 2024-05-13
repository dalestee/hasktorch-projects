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

module TextTreatment (textTreatment) where

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

textTreatment :: FilePath -> IO ()    
textTreatment filePath = do
  normalizeText filePath
  putStrLn "Text treatment done"