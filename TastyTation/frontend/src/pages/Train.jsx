import { useState, useEffect } from 'react'
import { Box, Card, CardContent, Typography, Button } from '@mui/material'
import { Hub, AutoAwesome } from '@mui/icons-material'
import ErrorInfoSnackbar from './components/ErrorInfoSnackbar'
import TrainingStatus from './components/TrainingStatus'

function ModelRow({ modelName, onDownload }) {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1, borderBottom: '1px solid #ddd', 
            borderRadius: 2, mb: 2, backgroundColor: '#f5f5f5', '&:hover': { backgroundColor: '#e0e0e0' } }}
        >
            <Typography variant="body1">{modelName}</Typography>
            <Button variant="contained" color="primary" onClick={() => onDownload(modelName)}>Download</Button>
        </Box>
    )
}

export default function Train() {
    const [error, setError] = useState('')
    const [training, setTraining] = useState(false)
    const [models, setModels] = useState([])

    function train() {
        setError('')
        setTraining(true)

        fetch('/api/train-detection-model', {
            'method': 'GET'
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to train model')
                }
                return response.json()
            })
            .then((data) => {
                setTraining(false)
            })
            .catch((error) => {
                setError(error.message)
                setTraining(false)
            })
    }

    useEffect(() => {
        getModels()
    }, [])

    function getModels() {
        fetch('/api/get-detection-models', {
            method: 'GET'
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to get models')
                }
                return response.json()
            })
            .then((data) => {
                setModels(data['models'])
            })
            .catch((error) => {
                setError('Failed to get models')
            })
    }

    function downloadModel(model='latest') {
        setError('')

        // If model is latest, download is requested after training, so need to refresh models list and get the latest model
        if (model === 'latest') {
            getModels()
            model = models[0]
        }

        const formData = new FormData()
        formData.append('model', model)

        fetch('/api/get-detection-model', {
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
                a.download = `${model}.pt`
                document.body.appendChild(a)
                a.click()
                window.URL.revokeObjectURL(url)
            })
            .catch((error) => {
                setError('Failed to download model')
            })
    }

    return (
        <Box sx={{ width: '95%', maxWidth: 1200, margin: '0 auto' }}>
            <ErrorInfoSnackbar error={error} setError={setError} info={training} infoMessage='Training new model...'/>

            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Hub sx={{ mr: 1 }} />
                            <Typography variant='h6'>Train</Typography>
                        </Box>
                        <Button variant='contained' color='primary' startIcon={<AutoAwesome />} onClick={train} disabled={training}>Train</Button>
                    </Box>

                    <TrainingStatus route={'detection'} downloadModel={downloadModel} setTraining={setTraining} />
                    
                    <Typography variant='h6' sx={{ mb: 2 }}>Models</Typography>
                    <Box>
                        {models.map((model) => (
                            <ModelRow key={model} modelName={model} onDownload={downloadModel} />
                        ))}
                    </Box>
                </CardContent>
            </Card>
        </Box>
    )
}