import React, { useEffect, useState, useRef } from 'react'
import { Box, Button, Card, CardContent, Typography, Snackbar, Alert } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { io } from 'socket.io-client'

import ErrorInfoSnackbar from './components/ErrorInfoSnackbar'
import AnnotationEditor from './bbox components/AnnotationEditor'
import AnnotatedImage from './bbox components/AnnotatedImage'

function NewAnnotations() {
    const [annotations, setAnnotations] = useState([])
    const [annotationStatus, setAnnotationStatus] = useState('LOADING')
    const [downloading, setDownloading] = useState(false)
    const [newAnnotationClasses, setNewAnnotationClasses] = useState([])
    const [currentIndex, setCurrentIndex] = useState(-1)
    const [error, setError] = useState('')
    const socketRef = useRef(null)

    const navigate = useNavigate()

    // Fetch annotations and setup socket subscriber
    useEffect(() => {
        fetchAnnotations()

        socketRef.current = io('http://localhost:5000', {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: 10,
            reconnectionDelay: 1000
        })

        function handleSocketData(data) {
            setAnnotations(data.annotations)
            setAnnotationStatus(data.status)
            setNewAnnotationClasses(data.new_annotation_classes)
        }

        socketRef.current.on('annotation_status', handleSocketData)

        return () => {
            if (socketRef.current) {
                socketRef.current.off('annotation_status', handleSocketData)
                socketRef.current.disconnect()
            }
        }
    }, [])

    // Fetch annotations
    async function fetchAnnotations() {
        try {
            const response = await fetch('/api/get-annotations')
            const data = await response.json()
            setAnnotations(data.annotations)
            setAnnotationStatus(data.status)
            setNewAnnotationClasses(data.new_annotation_classes)
        } catch (error) {
            setError(error.message)
        }
    }
    
    // Disable scrolling when the annotation editor is open
    useEffect(() => {
        if (currentIndex !== -1) {
            document.body.style.overflow = 'hidden'
        } else {
            document.body.style.overflow = 'auto'
        }

        return () => {
            document.body.style.overflow = 'auto'
        }
    }, [currentIndex])

    // Update annotations when they are changed
    function onAnnotationsChange(updatedAnnotations, image) {
        // Send updated annotations to the server
        fetch('/api/edit-labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image,
                annotations: updatedAnnotations,
            })
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to update annotations')
                }
            })
            .catch((error) => {
                setError(error.message)
                return
            })
        
        const updatedAnnotationsArray = annotations.map((item) => {
            // Match annotations with image
            if (item.image_path === image) {
                return { ...item, annotations: updatedAnnotations }
            }
            return item
        })
    
        const updatedNewAnnotationClasses = [...newAnnotationClasses]
        for (const annotation of updatedAnnotations) {
            // Find the index of the class in the current array
            const existingIndex = updatedNewAnnotationClasses.findIndex(
                (classItem) => classItem.id === annotation.class_id
            )
            if (existingIndex !== -1) {
                const [removedClass] = updatedNewAnnotationClasses.splice(existingIndex, 1)
                updatedNewAnnotationClasses.unshift(removedClass)
            }
        }
        
        setNewAnnotationClasses(updatedNewAnnotationClasses)
        setAnnotations(updatedAnnotationsArray)
    }

    function addAnnotation() {
        fetch('api/add-annotations', {
            method: 'POST',
        })
            .then((response) => {
                if (!response.ok) {
                    setError('Unable to add annotations')
                    return
                } else {
                    navigate('/train')
                }
            })
    }

    function onDeleteImage(imagePath) {
        const formData = new FormData()
        formData.append('imagePath', imagePath)

        fetch('/api/delete-image', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to delete image')
                } else {
                    if (currentIndex >= annotations.length - 1) {
                        setCurrentIndex(currentIndex - 1)
                    }
                    setAnnotations(annotations.filter((item) => item.image_path !== imagePath))
                }
            })
            .catch((error) => {
                setError(error.message)
            })
    }

    function downloadCroppedItems() {
        setDownloading(true)
        fetch('/api/download-cropped-items')
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to download cropped items')
                }
                return response.blob()
            })
            .then((blob) => {
                const url = window.URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'cropped_items.zip'
                a.click()
                window.URL.revokeObjectURL(url)
                setDownloading(false)
            })
            .catch((error) => {
                setError(error.message)
                setDownloading(false)
            })
    }

    return (
        <>
            <ErrorInfoSnackbar error={error} setError={setError} info={downloading} infoMessage={<>Downloading Cropped Items...<br />Please Do Not Close This Page</>} />

            {/* List of images */}
            <Box sx={{ p: 4, pt: 2 }}>
                {/* Header */}
                <Box sx={{ width: '95%', maxWidth: 1200, margin: '0 auto', mb: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Button onClick={fetchAnnotations} variant='contained'>
                            Refresh
                        </Button>
                        <Typography variant='body1'>Status: {annotationStatus}</Typography>
                    </Box>
                    {annotationStatus === 'DONE' && (
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <Button variant='contained' onClick={downloadCroppedItems}>Download Cropped Items</Button>
                            <Button variant='contained' onClick={addAnnotation}>Add Annotations</Button>
                        </Box>
                    )}
                </Box>
                {/* Images */}
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: '1fr 1fr 1fr' }, gap: 4 }}>
                    {annotations.map((item, index) => (
                        <Card key={item.image_path} sx={{ overflow: 'hidden' }}>
                            <Box onClick={() => setCurrentIndex(index)}>
                                <AnnotatedImage item={item} />
                            </Box>
                            <CardContent>
                                <Typography variant='body2'>
                                    {item.image_path.split('/').pop()} ({item.annotations.length} annotations)
                                </Typography>
                            </CardContent>
                        </Card>
                    ))}
                </Box>
            </Box>
            
            {/* Annotation Editor */}
            {currentIndex !== -1 && (
                <AnnotationEditor
                    annotations={annotations}
                    currentIndex={currentIndex}
                    setCurrentIndex={setCurrentIndex}
                    annotationClasses={newAnnotationClasses}
                    onAnnotationsChange={onAnnotationsChange}
                    onDeleteImage={onDeleteImage}
                />
            )}
        </>
    )
}

export default NewAnnotations