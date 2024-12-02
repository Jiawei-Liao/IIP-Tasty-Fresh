import React, { useEffect, useState } from 'react'
import { Box, Button, Card, CardContent, Typography } from '@mui/material'
import { io } from 'socket.io-client'

import AnnotationEditor from './components/AnnotationEditor'
import AnnotatedImage from './components/AnnotatedImage'

function NewAnnotations() {
    const [annotations, setAnnotations] = useState([])
    const [annotationStatus, setAnnotationStatus] = useState('LOADING')
    const [newAnnotationClasses, setNewAnnotationClasses] = useState([])
    const [currentIndex, setCurrentIndex] = useState(-1)

    // Fetch annotations and setup socket subscriber
    useEffect(() => {
        fetchAnnotations()

        const socket = io('http://localhost:5000')
        socket.on('annotation_status', (data) => {
            setAnnotationStatus(data.status)
            setAnnotations(data.annotations)
            setNewAnnotationClasses(data.new_annotation_classes)
        })

        return () => socket.disconnect()
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
            console.error('Error fetching annotations:', error)
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
                    console.error('Error updating annotations:', response.statusText)
                    return
                }
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
    
    return (
        <>
            {/* List of images */}
            <Box sx={{ p: 4 }}>
                {/* Header */}
                <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Button onClick={fetchAnnotations} variant="contained">
                        Refresh
                    </Button>
                    <Typography variant="body1">Status: {annotationStatus}</Typography>
                </Box>
                {/* Images */}
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: '1fr 1fr 1fr' }, gap: 4 }}>
                    {annotations.map((item, index) => (
                        <Card key={item.image_path} sx={{ overflow: 'hidden' }}>
                            <Box onClick={() => setCurrentIndex(index)}>
                                <AnnotatedImage item={item} />
                            </Box>
                            <CardContent>
                                <Typography variant="body2">
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
                    newAnnotationClasses={newAnnotationClasses}
                    onAnnotationsChange={onAnnotationsChange}
                />
            )}
        </>
    )
}

export default NewAnnotations