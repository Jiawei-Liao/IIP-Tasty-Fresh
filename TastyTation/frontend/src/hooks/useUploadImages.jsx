import { useState, useCallback } from 'react'

const CAPTURE_RATE = 1 // Seconds between frame captures

/**
 * Helper hook to handle image or video uploads to convert them into image objects
 * @param {React.Dispatch<String>} setError: Function to set error message
 * @param {React.Dispatch<Boolean>} setLoading: Function to set loading state
 * @returns { images, setImages, handleImageUpload }: Images state, function to set images state, function to handle image uploads
 */
export function useUploadImages({ setError, setLoading }) {
    const [images, setImages] = useState(new Map())

    // Creates an image object, optionally with a tier
    function createImageObject(src, file, tier=null) {
        const imageObject = {
            src,
            file,
            name: file.name
        }

        if (tier) {
            imageObject.tier = tier
        }

        return imageObject
    }

    // Gets uploaded images/videos and adds it to the images state
    // tier is used in Upload.jsx for auto annotation
    const handleImageUpload = useCallback(async (event, tier=null) => {
        try {
            setLoading(true)
            const files = Array.from(event.target.files)
    
            const processFile = async (file) => {
                if (file.type.startsWith('image/')) {
                    await processImageFile(file, tier)
                } else if (file.type === 'video/mp4') {
                    await extractFrames(file, tier)
                }
            }
    
            await Promise.all(files.map(processFile))
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false)
            event.target.value = ''
        }
    }, [setError, setLoading])

    function processImageFile(file, tier=null) {
        return new Promise((resolve) => {
            const reader = new FileReader()
            reader.onload = (e) => {
                setImages((prevImages) => {
                    const newImages = new Map(prevImages)
                    newImages.set(file.name, createImageObject(e.target.result, file, tier))
                    return newImages
                })
                resolve()
            }
            reader.readAsDataURL(file)
        })
    }

    function extractFrames(file, tier=null) {
        return new Promise((resolve, reject) => {
            // Create video element
            const video = document.createElement('video')
            video.src = URL.createObjectURL(file)
            video.crossOrigin = 'anonymous'
            
            // Create canvas element for frame capture
            const canvas = document.createElement('canvas')
            const ctx = canvas.getContext('2d')

            const processedFrames = []
    
            video.addEventListener('loadedmetadata', () => {
                const duration = Math.floor(video.duration)
                let currentSecond = 0
    
                function captureNextFrame() {
                    // Stop condition
                    if (currentSecond >= duration) {
                        URL.revokeObjectURL(video.src)
                        resolve(processedFrames)
                        return
                    }
    
                    // Set video time
                    video.currentTime = currentSecond
    
                    // Wait for seeked event
                    video.onseeked = () => {
                        return new Promise((frameResolve) => {
                            canvas.width = video.videoWidth
                            canvas.height = video.videoHeight
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    
                            canvas.toBlob((blob) => {
                                if (!blob) {
                                    console.warn(`Could not create blob for second ${currentSecond}`)
                                    currentSecond += CAPTURE_RATE
                                    captureNextFrame()
                                    frameResolve()
                                    return
                                }
    
                                // Pad the seconds to ensure consistent naming
                                const paddedSecond = currentSecond.toString().padStart(2, '0')
                                const frameFile = new File(
                                    [blob], 
                                    `${file.name}-frame-${paddedSecond}.png`, 
                                    { type: 'image/png' }
                                )
    
                                const reader = new FileReader()
    
                                reader.onload = (e) => {
                                    // Update images state
                                    setImages((prevImages) => {
                                        const newImages = new Map(prevImages)
                                        newImages.set(frameFile.name, createImageObject(e.target.result, frameFile, tier))
                                        return newImages
                                    })
    
                                    // Track processed frames if needed
                                    processedFrames.push(frameFile)
    
                                    // Move to next frame
                                    currentSecond += CAPTURE_RATE
                                    captureNextFrame()
                                    frameResolve()
                                }
    
                                reader.onerror = (error) => {
                                    console.warn('Error reading frame', error)
                                    currentSecond += CAPTURE_RATE
                                    captureNextFrame()
                                    frameResolve()
                                }
    
                                reader.readAsDataURL(frameFile)
                            })
                        })
                    }
                }
    
                // Start frame capture
                captureNextFrame()
            })

            video.onerror = (error) => {
                console.warn('Video processing error', error)
                reject(error)
            }
        })
    }

    return { images, setImages, handleImageUpload }
}