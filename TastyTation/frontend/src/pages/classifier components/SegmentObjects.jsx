import { useState, useCallback } from 'react'
import { Box, Typography, Button } from '@mui/material'
import { Start } from '@mui/icons-material'

import ErrorInfoSnackbar from '../components/ErrorInfoSnackbar'
import UploadImages from '../components/UploadImages'
import { useUploadImages } from '../../hooks/useUploadImages'
import ImagesList from '../components/ImagesList'

export default function SegmentObjects() {
    // Loading state for uploaded images
    const [loading, setLoading] = useState(false)

    // Error state
    const [error, setError] = useState(null)

    // Backend segmenting status
    const [segmenting, setSegmenting] = useState(false)

    // Store uploaded images
    const { images, setImages, handleImageUpload } = useUploadImages({ setError: setError, setLoading: setLoading })

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
                    <UploadImages handleImageUpload={handleImageUpload} disabled={loading} />
                    <Button variant='contained' onClick={segmentObjects} endIcon={<Start />} disabled={segmenting || loading || images.size == 0}>Segment Objects</Button>
                </Box>
                
                {/* Uploaded Images */}
                <ImagesList loading={loading} images={images} handleDeleteImage={handleDeleteImage} elevation={0} />
            </Box>
        </>
    )
}