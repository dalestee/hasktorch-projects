-- Download the data from the link below and put it in the data folder
-- https://drive.google.com/file/d/1p5Us05eBPO5DMKF6jeb0DZHg5m_cmqO5/view
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

module Word2vec (word2vec) where

import TextTreatment (textTreatment)

word2vec :: IO ()
word2vec = do
--   textTreatment "app/word2vec/data/review-texts.txt"
--   putStrLn "Word2vec done"

