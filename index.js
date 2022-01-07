import { NeuralNetworkGPU } from './brain.cjs'
import { readdir } from 'fs/promises'
import jimp from 'jimp'
import { dirname } from 'dirname-filename-esm'
import { join } from 'path'

const __dirname = dirname(import.meta)

const imageSize = 20

const getCompressedImage = async path => {
  const image = await jimp.read(path)
    const compressedImage = image.resize(imageSize, imageSize)
    return compressedImage
}

;(async () => {
  const net = new NeuralNetworkGPU()
  const trainingDir = join(__dirname, './train')
  const colors = await readdir(trainingDir)
  const trainingData = (await Promise.all(colors.map(async color => {
    const colorDir = join(trainingDir, color)
    const images = await readdir(colorDir)
    return await Promise.all(images.map(async imageName => {
      const compressedImage = await getCompressedImage(join(colorDir, imageName))
      return { 
        input: compressedImage.bitmap.data, 
        output: {
          [color]: 1
        }
      }
    }))
  }))).flat()

  console.time('Training')
  net.train(trainingData, { log: true })
  console.timeEnd('Training')

  console.time('Running')
  const runDir = join(__dirname, './run')
  const results = (await Promise.all(colors.map(async color => {
    const colorDir = join(runDir, color)
    const images = await readdir(colorDir)
    return await Promise.all(images.map(async imageName => {
      const imagePath = join(colorDir, imageName)
      const compressedImage = await getCompressedImage(imagePath)
      const result = net.run(compressedImage.bitmap.data)
      return {
        result,
        color,
        imageName
      }
    }))
  }))).flat()
  console.timeEnd('Running')
  console.log(results)

})().catch(console.error)

// jimp.read('./train/blue/download.jpg').then(image => {
//   image = image.scaleToFit(50, 100)
//   image.write('./t.jpg')

//   console.time('Training')
//   net.train([
//     {input: image.bitmap.data, output: { blue: 1 } }
//   ])
//   console.timeEnd('Training')

//   console.time('Running')
//   const result = net.run(image.bitmap.data)
//   console.timeEnd('Running')
//   console.log(result)
// })


// net.train([

// ])