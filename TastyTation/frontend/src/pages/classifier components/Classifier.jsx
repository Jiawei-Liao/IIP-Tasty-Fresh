import { useState, useCallback } from 'react'
import { Typography, Box, Button, TextField } from '@mui/material'
import { AutoAwesome, CloudDownload, CloudUpload } from '@mui/icons-material'

import ErrorInfoSnackbar from '../components/ErrorInfoSnackbar'
import UploadImages from '../components/UploadImages'
import ImagesList from '../components/ImagesList'
import TrainingStatus from '../components/TrainingStatus'
import { useUploadImages } from '../../hooks/useUploadImages'
import ViewClassesModel from './ViewClassesModal'

export default function Classifier({ route }) {
    const [loading, setLoading] = useState(false)
    const [training, setTraining] = useState(false)
    const [error, setError] = useState('')
    const [className, setClassName] = useState('')
    const { images, setImages, handleImageUpload } = useUploadImages({ setError: setError, setLoading: setLoading })
    const [classes, setClasses] = useState({})

    const handleDeleteImage = useCallback((imageName) => {
        // Removes selected image
        setImages((prevImages) => {
            const newImages = new Map(prevImages)
            newImages.delete(imageName)
            return newImages
        })
    }, [])

    function addImages() {
        // Start auto annotation process, uploading images
        setError(null)
        setLoading(true)

        // Validate upload
        if (images.size === 0) {
            setError('No images uploaded')
            setLoading(false)
            return
        }

        if (className === '') {
            setError('Class name cannot be empty')
            setLoading(false)
            return
        }
    
        // Prepare data for upload
        const formData = new FormData()
        formData.append('classifierName', route)
        formData.append('className', className)
        Array.from(images.values()).forEach((image) => {
            formData.append('images', image.file, JSON.stringify({ name: image.name }))
        })
    
        // Send data to backend
        fetch('/api/add-classifier-images', {
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
                setImages(new Map())
            })
            .catch((error) => {
                setLoading(false)
                setError(error.message)
            })
            .finally(() => {
                setLoading(false)
                setClassName('')
            })
    }

    function trainNewModel() {
        setError(null)
        setTraining(true)

        const formData = new FormData()
        formData.append('classifierName', route)

        fetch('/api/train-classifier', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to train model')
                }
                return response.json()
            })
            .catch((error) => {
                setError(error.message)
                setTraining(false)
            })
    }

    function downloadModel() {
        setError(null)
        setLoading(true)

        const formData = new FormData()
        formData.append('classifierName', route)

        fetch('/api/get-classifier-model', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to download model')
                }
                return response.blob()
            })
            .then((blob) => {
                const url = window.URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `${route}.pt`
                document.body.appendChild(a)
                a.click()
                window.URL.revokeObjectURL(url)
            })
            .catch((error) => {
                setLoading(false)
                setError('Failed to download model')
            })
            .finally(() => {
                setLoading(false)
            })
    }

    function viewClasses() {
        const formData = new FormData()
        formData.append('classifierName', route)

        fetch('/api/view-classifier-classes', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to view classes')
                }
                return response.json()
            })
            .then((data) => {
                if (data && data.classes && Object.keys(data.classes).length > 0) {
                    setClasses(data.classes)
                } else {
                    setError('No classes found')
                }
            })
            .catch((error) => {
                setError(error.message)
            })
    }

    return (
        <>
            <ErrorInfoSnackbar error={error} setError={setError} info={training} infoMessage='Training new model...' />

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant='h4'>{route}</Typography>
                    <Button variant='contained' onClick={viewClasses}>View Classes</Button>
                </Box>

                <Typography variant='caption' style={{ display: 'flex', justifyContent: 'center', textAlign: 'center' }}>
                    Upload images for a single class, specify its name (new or existing), and add them to the dataset. Repeat this process for every new class you want to create.
                </Typography>

                <TrainingStatus route={route} downloadModel={downloadModel} setTraining={setTraining} />

                <Box sx={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 4 }}>
                    <UploadImages handleImageUpload={handleImageUpload} disabled={loading} video={false} />
                    <Button variant='contained' startIcon={<CloudUpload />} disabled={!(images.size > 0 && className) || loading} onClick={addImages}>Add Images to Dataset</Button>
                    <Button variant='contained' startIcon={<AutoAwesome />} disabled={training} onClick={trainNewModel}>Train New Model</Button>
                    <Button variant='contained' startIcon={<CloudDownload />} disabled={training} onClick={downloadModel}>Download Model</Button>
                </Box>
                <TextField label='Class Name' value={className} onChange={(e) => setClassName(e.target.value)} fullWidth /> 
                <ImagesList images={images} handleDeleteImage={handleDeleteImage} elevation={0} />
            </Box>

            {Object.keys(classes).length > 0 && <ViewClassesModel classes={classes} setClasses={setClasses} classifierName={route} setError={setError} />}
        </>
    )
}