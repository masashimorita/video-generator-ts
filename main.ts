import * as fs from 'fs';
import * as path from 'path';
import fetch from 'node-fetch';
import dotenv from "dotenv";
import { z } from "zod";
import { ChatOpenAI } from '@langchain/openai';
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const ffmpeg = require('fluent-ffmpeg');

dotenv.config();

const {
  OPENAI_API_KEY,
  UNSPLASH_ACCESS_KEY,
  JAMENDO_CLIENT_ID
} = process.env;

if (!OPENAI_API_KEY) {
  console.error("Error: OPENAI_API_KEY 環境変数が設定されていません。");
  process.exit(1);
}
if (!UNSPLASH_ACCESS_KEY) {
  console.error("Error: UNSPLASH_ACCESS_KEY 環境変数が設定されていません。");
  process.exit(1);
}
if (!JAMENDO_CLIENT_ID) {
  console.error("Error: JAMENDO_CLIENT_ID 環境変数が設定されていません。");
  process.exit(1);
}

// OpenAI の初期化
const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
})

/**
 * 入力テキストから要約を生成する関数
 */
async function generateSummary(text: string): Promise<string> {
  try {
    const outputSchema = z.object({
      summary: z.string().describe("Summary of a text"),
    });
  
    const promptTemplate = ChatPromptTemplate.fromTemplate(`
      以下のテキストを日本語で要約してください:
      {text}
      要約はsummaryとして返却してください。
    `);
  
    const model = llm.withStructuredOutput(outputSchema);
  
    const chain = RunnableSequence.from([
      promptTemplate,
      model,
    ]);

    const result = await chain.invoke({ text });
    return result.summary;
  } catch (error: any) {
    throw new Error("要約生成中にエラーが発生しました: " + error);
  }
}

/**
 * 要約文から画像用キーワードを抽出する関数
 */
async function extractImageKeyword(summary: string): Promise<string> {
  try {
    const outputSchema = z.object({
      keyword: z.string().describe('most relevant keyword of summary'),
    });
  
    const promptTemplate = ChatPromptTemplate.fromTemplate(`
      以下の要約文から、画像取得に最も適したキーワードを1語だけ返してください。要約: 
      {summary}
      キーワードはkeywordとして返却してください。
    `);
  
    const model = llm.withStructuredOutput(outputSchema);
  
    const chain = RunnableSequence.from([
      promptTemplate,
      model,
    ]);

    const result = await chain.invoke({ summary });
    return result.keyword.trim();
  } catch (error: any) {
    throw new Error("画像キーワード抽出中にエラーが発生しました: " + error);
  }
}

/**
 * 要約文から音楽用キーワードを抽出する関数
 */
async function extractMusicKeyword(summary: string): Promise<string> {
  try {
    const outputSchema = z.object({
      keyword: z.string().describe('most relevant keyword of summary'),
    });
  
    const promptTemplate = ChatPromptTemplate.fromTemplate(`
      以下の要約文から、音楽取得に最も適したキーワードを1語だけ返してください。
      要約: {summary}
      キーワードはkeywordとして返却してください。
    `);
  
    const model = llm.withStructuredOutput(outputSchema);
  
    const chain = RunnableSequence.from([
      promptTemplate,
      model,
    ]);

    const result = await chain.invoke({ summary });
    return result.keyword.trim();
  } catch (error: any) {
    throw new Error("音楽キーワード抽出中にエラーが発生しました: " + error);
  }
}

/**
 * Unsplash API を用いて、指定キーワードに基づいた画像を取得しローカルに保存する関数
 */
async function fetchRelatedImage(keyword: string, savePath: string = "related_image.jpg"): Promise<string> {
  const url = `https://api.unsplash.com/photos/random?query=${encodeURIComponent(keyword)}&client_id=${UNSPLASH_ACCESS_KEY}`;
  const response = await fetch(url);
  if (response.ok) {
    const data = await response.json() as any;
    const imageUrl: string = data.urls.regular;
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      throw new Error("画像のダウンロードに失敗しました。");
    }
    const buffer = await imageResponse.buffer();
    fs.writeFileSync(savePath, buffer);
    return savePath;
  } else {
    throw new Error("Unsplashから画像取得中にエラーが発生しました: " + await response.text());
  }
}

/**
 * Jamendo API を用いて、指定キーワードに基づいた音楽を取得しローカルに保存する関数
 */
async function fetchRelatedMusic(keyword: string, savePath: string = "related_music.mp3"): Promise<string> {
  const url = `https://api.jamendo.com/v3.0/tracks/?client_id=${JAMENDO_CLIENT_ID}&format=json&limit=1&search=${encodeURIComponent(keyword)}`;
  const response = await fetch(url);
  if (response.ok) {
    const data = await response.json() as any;
    if (data.results && data.results.length > 0) {
      const musicUrl: string = data.results[0].audio;
      const musicResponse = await fetch(musicUrl);
      if (!musicResponse.ok) {
        throw new Error("音楽ファイルのダウンロードに失敗しました。");
      }
      const buffer = await musicResponse.buffer();
      fs.writeFileSync(savePath, buffer);
      return savePath;
    } else {
      throw new Error("Jamendoから音楽が見つかりませんでした。");
    }
  } else {
    throw new Error("Jamendo API 呼び出し中にエラーが発生しました: " + await response.text());
  }
}

/**
 * fluent-ffmpeg を利用して、画像、テキスト（要約）、音楽を合成しビデオを生成する関数
 */
function createVideo(
  summaryText: string,
  imagePath: string,
  musicPath: string,
  outputPath: string = "output_video.mp4",
  duration: number = 20
): Promise<void> {
  return new Promise((resolve, reject) => {
    // フォントファイルのパス（環境に合わせて調整してください）
    const fontPath = path.join(__dirname, 'Arial.ttf');
    // ffmpeg 用にテキストをエスケープ
    const escapedText = summaryText.replace(/:/g, '\\:').replace(/'/g, "\\'");
    
    ffmpeg()
      .addInput(imagePath)
      .loop(duration)
      .addInput(musicPath)
      .videoFilters({
        filter: 'drawtext',
        options: {
          fontfile: fontPath,
          text: escapedText,
          fontsize: 24,
          fontcolor: 'white',
          x: '(w-text_w)/2',
          y: '(h-text_h)/2',
          box: 1,
          boxcolor: 'black@0.5'
        }
      })
      .duration(duration)
      .outputOptions('-c:v libx264', '-pix_fmt yuv420p', '-c:a aac')
      .save(outputPath)
      .on('end', () => {
        console.log("ビデオが生成されました:", outputPath);
        resolve();
      })
      .on('error', (err: any) => {
        console.error("ビデオ生成中にエラーが発生しました:", err);
        reject(err);
      });
  });
}

function getText() {
  return fs.readFileSync('./sample.txt','utf8');
}

/**
 * メイン処理：要約生成、キーワード抽出、素材取得、ビデオ生成を統合
 */
async function main(): Promise<void> {
  try {
    const inputText = getText();
    console.log("要約を生成中...");
    const summary = await generateSummary(inputText);
    console.log("生成された要約:");
    console.log(summary);

    // 要約から画像用および音楽用のキーワードを抽出
    const imageKeyword = await extractImageKeyword(summary);
    console.log("画像用キーワード:", imageKeyword);
    const musicKeyword = await extractMusicKeyword(summary);
    console.log("音楽用キーワード:", musicKeyword);

    // キーワードに基づいて画像・音楽を取得（エラー発生時はフォールバック素材を利用）
    let imagePath: string;
    try {
      imagePath = await fetchRelatedImage(imageKeyword, "related_image.jpg");
      console.log(imagePath);
    } catch (error) {
      console.error("画像取得エラー:", error);
      imagePath = "default_background.jpg";
    }
    let musicPath: string;
    try {
      musicPath = await fetchRelatedMusic(musicKeyword, "related_music.mp3");
    } catch (error) {
      console.error("音楽取得エラー:", error);
      musicPath = "default_music.mp3";
    }

    // ビデオ生成
    await createVideo(summary, imagePath, musicPath, "output_video.mp4", 20);
  } catch (error) {
    console.error("エラー:", error);
  }
}

main();
