import { useState, useCallback } from 'react'
import { Box, Button, Typography, TextField, ToggleButton, ToggleButtonGroup, Card, CardContent } from '@mui/material'
import { FileUpload, Start } from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import ErrorInfoSnackbar from './components/ErrorInfoSnackbar'
import UploadImages from './components/UploadImages'
import { useUploadImages } from '../hooks/useUploadImages'
import ImagesList from './components/ImagesList'

const TIERS = {
    TIER1: { name: 'Tier 1', description: 'Single item, no noise' },
    TIER2: { name: 'Tier 2', description: 'Multiple items, no noise OR single item, with noise' },
    TIER3: { name: 'Tier 3', description: 'Multiple items, with noise' }
}

export default function Upload() {
    const navigate = useNavigate()

    // Loading state for uploaded images
    const [loading, setLoading] = useState(false)

    // Upload settings
    const [uploadType, setUploadType] = useState('existing')
    const [className, setClassName] = useState('')

    // Error state
    const [error, setError] = useState(null)

    // Upload status
    const [uploading, setUploading] = useState(false)

    // Store uploaded images
    const { images, setImages, handleImageUpload } = useUploadImages({ setError: setError, setLoading: setLoading })

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
                <UploadImages handleImageUpload={(e) => handleImageUpload(e, tier)} key={tier.name} disabled={loading} buttonText={`Upload ${tier.name}`} />
            ))
        } else {
            return (
                <UploadImages handleImageUpload={(e) => handleImageUpload(e, TIERS.TIER3)} disabled={loading} />
            )
        }
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
            <ImagesList images={images} handleDeleteImage={handleDeleteImage} handleTierChange={handleTierChange} uploadType={uploadType} tiers={TIERS} />
        </Box>
    )
}
