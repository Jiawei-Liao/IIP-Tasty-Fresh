import { useState, useCallback } from 'react'
import { Box, Typography, Button, ImageList, ImageListItem, Card, CardContent, CircularProgress, IconButton, Snackbar, Alert } from '@mui/material'
import { Delete, CloudUpload, Start } from '@mui/icons-material'
import ErrorInfoSnackbar from '../components/ErrorInfoSnackbar'

export default function SegmentObjects() {
    // Store uploaded images
    const [images, setImages] = useState(new Map())

    // Loading state for uploaded images
    const [loading, setLoading] = useState(false)

    // Error state
    const [error, setError] = useState(null)

    // Backend segmenting status
    const [segmenting, setSegmenting] = useState(false)

    // Seconds between each capture
    const [captureRate, setCaptureRate] = useState(0.5)

    const handleImageUpload = useCallback(async (event) => {
        try {
            setLoading(true)
            const files = Array.from(event.target.files)
    
            const processFile = async (file) => {
                return new Promise((resolve) => {
                    if (file.type.startsWith('image/')) {
                        const reader = new FileReader()
                        reader.onload = (e) => {
                            setImages((prevImages) => {
                                const newImages = new Map(prevImages)
                                newImages.set(file.name, {
                                    src: e.target.result,
                                    file: file,
                                    name: file.name,
                                })
                                return newImages
                            })
                            resolve()
                        }
                        reader.readAsDataURL(file)
                    } else if (file.type === 'video/mp4') {
                        extractFrames(file).then(resolve)
                    }
                })
            }
    
            await Promise.all(files.map(processFile))
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false)
            event.target.value = ''
        }
    }, [])
        
    function extractFrames(file) {
        return new Promise((resolve, reject) => {
            // Create video element
            const video = document.createElement('video')
            video.src = URL.createObjectURL(file)
            video.crossOrigin = 'anonymous'
            
            // Create canvas element for frame capture
            const canvas = document.createElement('canvas')
            const ctx = canvas.getContext('2d')
    
            video.addEventListener('loadedmetadata', () => {
                const duration = Math.floor(video.duration)
                let currentSecond = 0
                const processedFrames = []
    
                const captureNextFrame = () => {
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
                                    currentSecond += captureRate
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
                                        newImages.set(frameFile.name, {
                                            src: e.target.result,
                                            file: frameFile,
                                            name: frameFile.name,
                                        })
                                        return newImages
                                    })
    
                                    // Track processed frames if needed
                                    processedFrames.push(frameFile)
    
                                    // Move to next frame
                                    currentSecond += captureRate
                                    captureNextFrame()
                                    frameResolve()
                                }
    
                                reader.onerror = (error) => {
                                    console.warn('Error reading frame', error)
                                    currentSecond += captureRate
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

    const handleDeleteImage = useCallback((imageName) => {
        // Removes selected image
        setImages((prevImages) => {
            const newImages = new Map(prevImages)
            newImages.delete(imageName)
            return newImages
        })
    }, [])

    function segmentObjects() {
        // Start auto annotation process, uploading images
        setError(null)
        setSegmenting(true)
        // Validate upload
        if (images.size === 0) {
            setError('No images uploaded')
            setSegmenting(false)
            return
        }

        // Prepare data for upload
        const formData = new FormData()
        Array.from(images.values()).forEach((image) => {
            formData.append('images', image.file, image.name)
        })

        // Send data to backend
        fetch('/api/segment-images', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Segmentation failed')
                }
                return response.blob()
            })
            .then((blob) => {
                const url = window.URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'segmented_images.zip'
                document.body.appendChild(a)
                a.click()
                window.URL.revokeObjectURL(url)
            })
            .catch((error) => {
                setSegmenting(false)
                setError('Segmentation failed')
            })
            .finally(() => {
                setImages(new Map())
                setSegmenting(false)
            })
    }

    return (
        <>
            {/* Error Snackbar */}
            <ErrorInfoSnackbar error={error} setError={setError} info={segmenting} infoMessage={<>Segmenting Objects...<br />Please Do Not Close This Page</>} />

            {/* Main Content */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography variant='h4'>Segment Objects</Typography>

                <Typography variant='caption' style={{ display: 'flex', justifyContent: 'center', textAlign: 'center' }}>Upload images or videos to segment for individual classifier datasets</Typography>

                <Box sx={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'}}>
                    <Box>
                        <input
                            accept='image/*,video/mp4'
                            style={{ display: 'none' }}
                            id='image-upload-button'
                            multiple
                            type='file'
                            onChange={(e) => handleImageUpload(e)}
                            disabled={loading}
                        />
                        <label htmlFor='image-upload-button'>
                            <Button
                                variant='contained'
                                component='span'
                                startIcon={<CloudUpload />}
                                disabled={segmenting || loading}
                                sx={{ width: '200px' }}
                            >
                                Upload Images
                            </Button>
                        </label>
                    </Box>
                    <Button variant='contained' onClick={segmentObjects} endIcon={<Start />} disabled={segmenting || loading}>Segment Objects</Button>
                </Box>
                
                {/* Uploaded Images */}
                <Card sx={{ mb: 3 }}>
                    <CardContent>
                        {/* Loading Spinner */}
                        {loading && (
                            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                                <CircularProgress />
                            </Box>
                        )}

                        {/* Image List */}
                        {images.size > 0 ? (
                            <ImageList sx={{ width: '100%' }} cols={3} rowHeight={300}>
                                {Array.from(images.values()).map((image) => (
                                    <ImageListItem
                                        key={image.name}
                                        sx={{
                                            position: 'relative',
                                            border: '1px solid #eee',
                                            borderRadius: 1,
                                            overflow: 'hidden',
                                        }}
                                    >
                                        {/* Image */}
                                        <img
                                            src={image.src}
                                            alt={image.name}
                                            loading='lazy'
                                            style={{ height: '240px', objectFit: 'cover' }}
                                        />
                                        {/* Delete button */}
                                        <IconButton
                                            sx={{
                                                position: 'absolute',
                                                top: 5,
                                                right: 5,
                                                backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                                '&:hover': {
                                                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                                                },
                                            }}
                                            onClick={() => handleDeleteImage(image.name)}
                                        >
                                            <Delete />
                                        </IconButton>
                                    </ImageListItem>
                                ))}
                            </ImageList>
                        ) : (
                            <Typography variant='body1' color='text.secondary' align='center'>
                                No Images Uploaded
                            </Typography>
                        )}
                    </CardContent>
                </Card>
            </Box>
        </>
    )
}