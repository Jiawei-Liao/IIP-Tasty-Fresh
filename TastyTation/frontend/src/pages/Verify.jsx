import { useState, useEffect, useRef } from 'react'
import { Box, Button, Typography, Card, CardContent, Alert, Snackbar } from '@mui/material'
import { io } from 'socket.io-client'

import VerificationAnnotationEditor from './bbox components/VerificationAnnotationEditor'
import AnnotatedImage from './bbox components/AnnotatedImage'

export default function Verify() {
    const [annotations, setAnnotations] = useState([])
    const [annotationStatus, setAnnotationStatus] = useState('LOADING')
    const [annotationClasses, setAnnotationClasses] = useState([])
    const [currentIndex, setCurrentIndex] = useState(-1)
    const [error, setError] = useState('')
    const [verifying, setVerifying] = useState(false)
    const socketRef = useRef(null)

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
            setAnnotations(data.inconsistent_annotations)
            setAnnotationStatus(data.status)
            setAnnotationClasses(data.annotation_classes)
        }

        socketRef.current.on('verification_status', handleSocketData)

        return () => {
            if (socketRef.current) {
                socketRef.current.off('verification_status', handleSocketData)
                socketRef.current.disconnect()
            }
        }
    }, [])

    // Fetch annotations
    async function fetchAnnotations() {
        try {
            const response = await fetch('/api/get-verification')
            const data = await response.json()
            setAnnotations(data.inconsistent_annotations)
            setAnnotationStatus(data.status)
            setAnnotationClasses(data.annotation_classes)

            if (data.status === 'STARTED') {
                setVerifying(true)
            } else {
                setVerifying(false)
            }
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
        
        const updatedAnnotationClasses = [...annotationClasses]
        for (const annotation of updatedAnnotations) {
            // Find the index of the class in the current array
            const existingIndex = updatedAnnotationClasses.findIndex(
                (classItem) => classItem.id === annotation.class_id
            )
            if (existingIndex !== -1) {
                const [removedClass] = updatedAnnotationClasses.splice(existingIndex, 1)
                updatedAnnotationClasses.unshift(removedClass)
            }
        }
        
        setAnnotationClasses(updatedAnnotationClasses)
        setAnnotations(updatedAnnotationsArray)
    }

    // Verify dataset
    function verifyDataset() {
        setVerifying(true)
        fetch('/api/verify-dataset', {
            method: 'POST',
        })
            .then((response) => {
                if (!response.ok) {
                    console.error('Error verifying dataset:', response.statusText)
                    setError('Error verifying dataset')
                    setVerifying(false)
                }
            })
    }

    // Resolve inconsistencies
    function resolveInconsistencies(imagePath) {
        const formData = new FormData()
        formData.append('image_path', imagePath)

        fetch('/api/resolve-inconsistency', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    console.error('Error resolving inconsistencies:', response.statusText)
                    setError('Error resolving inconsistencies')
                } else {
                    const updatedAnnotations = annotations.filter((item) => item.image_path !== imagePath)
                    setAnnotations(updatedAnnotations)
                }
            })
    }

    function updateRemoveLabel(imagePath, index) {
        const formData = new FormData()
        formData.append('image_path', imagePath)
        formData.append('label_index', index)
    
        fetch('/api/update-inconsistent-label', {
            method: 'POST',
            body: formData,
        })
            .then((response) => {
                if (!response.ok) {
                    console.error('Error removing label:', response.statusText)
                    setError('Error removing label')
                } else {
                    const updatedAnnotations = annotations.map((item) => {
                        if (item.image_path === imagePath) {
                            const updatedItem = { ...item }
                            
                            // Before removing the annotation, adjust the inconsistency indexes
                            const adjustedInconsistencyIndex = updatedItem.dataset_inconsistency_index.map(idx => 
                                idx > index ? idx - 1 : idx
                            ).filter(idx => idx !== index)
    
                            // Remove the annotation
                            updatedItem.annotations.splice(index, 1)
                            
                            // Update the inconsistency indexes
                            updatedItem.dataset_inconsistency_index = adjustedInconsistencyIndex
                            
                            return updatedItem
                        }
                        return item
                    })
    
                    setAnnotations(updatedAnnotations)
                }
            })
            .catch((error) => {
                console.error('Error during fetch:', error)
                setError('Error removing label')
            })
    }

    return (
        <>
            {/* Error Snackbar */}
            {error && (
                <Snackbar
                    open={Boolean(error)}
                    anchorOrigin={{ vertical: "top", horizontal: "center" }}
                >
                    <Alert severity="error" onClose={() => setError('')}>
                        {error}
                    </Alert>
                </Snackbar>
            )}
            {/* Verifying Status Snackbar */}
            {verifying && !error && (
                <Snackbar
                    open={verifying}
                    anchorOrigin={{ vertical: "top", horizontal: "center" }}
                >
                    <Alert severity="info">
                        Verifying Dataset...
                    </Alert>
                </Snackbar>
            )}
            {/* List of images */}
            <Box sx={{ p: 4, pt: 2 }}>
                {/* Header */}
                <Box sx={{ width: "95%", maxWidth: 1200, margin: "0 auto", display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Button onClick={fetchAnnotations} variant="contained">
                            Refresh
                        </Button>
                        <Typography variant="body1">Status: {annotationStatus}</Typography>
                    </Box>
                    <Box>
                        <Button variant="contained" onClick={verifyDataset} disabled={verifying}>Verify Dataset</Button>
                    </Box>
                </Box>
                <Typography variant="caption" style={{ display: 'flex', justifyContent: 'center', textAlign: 'center' }}>Validation will use latest trained model. If new data has been added, train a model before validating</Typography>
                {/* Images */}
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: '1fr 1fr 1fr' }, gap: 4, mt: 2 }}>
                    {annotations.map((item, index) => (
                        <Card key={item.image_path} sx={{ overflow: 'hidden' }}>
                            <Box onClick={() => setCurrentIndex(index)}>
                                <AnnotatedImage item={item} />
                            </Box>
                            <CardContent>
                                <Typography variant="body2">
                                    {item.inconsistencies
                                        .map(word => word.toLowerCase().replace(/^\w/, c => c.toUpperCase()))
                                        .join(', ')}
                                </Typography>
                            </CardContent>
                        </Card>
                    ))}
                </Box>
            </Box>

            {/* Annotation Editor */}
            {currentIndex !== -1 && (
                <VerificationAnnotationEditor
                    annotations={annotations}
                    currentIndex={currentIndex}
                    setCurrentIndex={setCurrentIndex}
                    annotationClasses={annotationClasses}
                    onAnnotationsChange={onAnnotationsChange}
                    resolveInconsistencies={resolveInconsistencies}
                    updateRemoveLabel={updateRemoveLabel}
                />
            )}
        </>
    )
}