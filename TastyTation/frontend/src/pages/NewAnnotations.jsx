import React, { useEffect, useState } from 'react'
import { Box, Button, Card, CardContent, Typography } from '@mui/material'
import { io } from 'socket.io-client'

import AnnotationEditor from './components/AnnotationEditor'
import AnnotatedImage from './components/AnnotatedImage'

const NewAnnotations = () => {
    const [annotations, setAnnotations] = useState([])
    const [annotationStatus, setAnnotationStatus] = useState('LOADING')
    const [currentIndex, setCurrentIndex] = useState(-1)

    // Fetch annotations and setup socket subscriber
    useEffect(() => {
        fetchAnnotations()

        const socket = io('http://localhost:5000')
        socket.on('annotation_status', (data) => {
            setAnnotationStatus(data.status)
            setAnnotations(data.annotations)
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
        } catch (error) {
            console.error('Error fetching annotations:', error)
        }
    }
    console.log(annotations)
    return (
        <>
            <Box sx={{ p: 4 }}>
                {/* Header */}
                <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Button onClick={fetchAnnotations} variant="contained">
                        Refresh
                    </Button>
                    <Typography variant="body1">Status: {annotationStatus}</Typography>
                </Box>
                {/* Images */}
                <Box
                    sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: '1fr 1fr 1fr' },
                        gap: 4,
                    }}
                >
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
                />
            )}
        </>
    )
}

export default NewAnnotations