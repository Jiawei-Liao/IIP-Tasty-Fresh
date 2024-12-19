import { useState, useCallback } from 'react'
import { Box, Button, IconButton, ImageList, ImageListItem, Typography, TextField, CircularProgress, Select, MenuItem, FormControl, ToggleButton, ToggleButtonGroup, Card, CardContent, Tooltip } from '@mui/material'
import { Delete, CloudUpload, FileUpload, Start } from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import ErrorInfoSnackbar from './components/ErrorInfoSnackbar'

const TIERS = {
    TIER1: { name: 'Tier 1', description: 'Single item, no noise' },
    TIER2: { name: 'Tier 2', description: 'Multiple items, no noise OR single item, with noise' },
    TIER3: { name: 'Tier 3', description: 'Multiple items, with noise' }
}

export default function Upload() {
    const navigate = useNavigate()

    // Store uploaded images
    const [images, setImages] = useState(new Map())

    // Loading state for uploaded images
    const [loading, setLoading] = useState(false)

    // Upload settings
    const [uploadType, setUploadType] = useState('existing')
    const [className, setClassName] = useState('')

    // Error state
    const [error, setError] = useState(null)

    // Upload status
    const [uploading, setUploading] = useState(false)

    // Seconds between each capture
    const [captureRate, setCaptureRate] = useState(0.5)

    function startAutoAnnotation() {
        // Start auto annotation process, uploading images
        setError(null)
        setUploading(true)

        // Validate upload
        if (images.size === 0) {
            setError('No images uploaded')
            setUploading(false)
            return
        }

        if (uploadType === 'new' && className === '') {
            setError('Class name cannot be empty')
            setUploading(false)
            return
        }

        // Prepare data for upload
        const formData = new FormData()
        formData.append('uploadType', uploadType)
        formData.append('className', className)
        Array.from(images.values()).forEach((image) => {
            formData.append('images', image.file, JSON.stringify({ name: image.name, tier: image.tier.name }))
        })

        // Send data to backend
        fetch('/api/annotate', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to upload images')
                }
                return response.json()
            })
            .then((data) => {
                navigate('/new-annotations')
            })
            .catch((error) => {
                setError(error.message)
            })
            .finally(() => {
                setUploading(false)
            })
    }
    
    function handleClassNameChange(e) {
        // Updates class name to uppercase and replace spaces with hyphens
        let updatedValue = e.target.value
        updatedValue = updatedValue.toUpperCase()
        updatedValue = updatedValue.replace(/ /g, '-')
        setClassName(updatedValue)
    }

    function renderUploadButtons() {
        // Render upload buttons based on upload type
        if (uploadType === 'new') {
            return Object.values(TIERS).map((tier) => (
                <Box key={tier.name}>
                    <input
                        accept='image/*,video/mp4'
                        style={{ display: 'none' }}
                        id={`image-upload-${tier.name}`}
                        multiple
                        type='file'
                        onChange={(e) => handleImageUpload(e, tier)}
                        disabled={loading}
                    />
                    <label htmlFor={`image-upload-${tier.name}`}>
                        <Tooltip title={tier.description} arrow>
                            <Button
                                variant='contained'
                                component='span'
                                startIcon={<CloudUpload />}
                                disabled={loading}
                                sx={{ width: '200px' }}
                            >
                                Upload {tier.name}
                            </Button>
                        </Tooltip>
                    </label>
                </Box>
            ))
        } else {
            return (
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
                            disabled={loading}
                            sx={{ width: '200px' }}
                        >
                            Upload Images
                        </Button>
                    </label>
                </Box>
            )
        }
    }

    const handleImageUpload = useCallback(async (event, defaultTier = TIERS.TIER3) => {
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
                                    tier: defaultTier,
                                })
                                return newImages
                            })
                            resolve()
                        }
                        reader.readAsDataURL(file)
                    } else if (file.type === 'video/mp4') {
                        extractFrames(file, defaultTier).then(resolve)
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
    
    function extractFrames(file, defaultTier) {
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
                                            tier: defaultTier,
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

    const handleTierChange = useCallback((imageKey, newTier) => {
        setImages((prevImages) => {
            const newImages = new Map(prevImages)
            const image = newImages.get(imageKey)
            if (image) {
                newImages.set(imageKey, { ...image, tier: newTier })
            }
            return newImages
        })
    }, [])

    return (
        <Box sx={{ width: '95%', maxWidth: 1200, margin: '0 auto' }}>
            <ErrorInfoSnackbar error={error} setError={setError} info={uploading} infoMessage='Uploading Images...'/>

            {/* Upload Type Toggle */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <FileUpload sx={{ mr: 1 }} />
                            <Typography variant='h6'>Upload Data</Typography>
                        </Box>
                        {/* New Button on the right */}
                        <Button
                            variant='contained'
                            color='primary'
                            endIcon={<Start />}
                            sx={{ width: 'auto' }}
                            disabled={uploading || images.size === 0 || loading}
                            onClick={startAutoAnnotation}
                        >
                            Start Auto Annotation
                        </Button>
                    </Box>
                    <Box sx={{ display: 'flex', mb: 2 }}>
                        <ToggleButtonGroup
                            value={uploadType}
                            exclusive
                            onChange={(e, newType) => newType && setUploadType(newType)}
                            fullWidth
                        >
                            <ToggleButton value='existing'>Add To Existing Classes</ToggleButton>
                            <ToggleButton value='new'>Add New Item</ToggleButton>
                        </ToggleButtonGroup>
                    </Box>
                    {uploadType === 'new' && (
                        <TextField
                            value={className}
                            onChange={handleClassNameChange}
                            fullWidth
                            variant='outlined'
                            label='Class Name'
                        />
                    )}
                </CardContent>
            </Card>

            {/* Image Upload Buttons */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4 }}>
                        {renderUploadButtons()}
                    </Box>
                </CardContent>
            </Card>

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
                                    {/* Tier Editer */}
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            top: 5,
                                            left: 5,
                                            p: 0.5,
                                            borderRadius: 1,
                                            cursor: 'pointer',
                                            fontSize: '0.8rem',
                                            textAlign: 'center',
                                        }}
                                    >
                                        {uploadType === 'new' && (
                                            <FormControl fullWidth size='small'>
                                                <Select
                                                    value={image.tier}
                                                    onChange={(e) =>
                                                        handleTierChange(
                                                            image.path || image.name,
                                                            e.target.value
                                                        )
                                                    }
                                                >
                                                    {Object.values(TIERS).map((tier) => (
                                                        <MenuItem key={tier.name} value={tier}>
                                                            {tier.name}
                                                        </MenuItem>
                                                    ))}
                                                </Select>
                                            </FormControl>
                                        )}
                                    </Box>
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
    )
}
