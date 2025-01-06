import { useEffect, useRef, useState } from 'react'
import { io } from 'socket.io-client'
import { Typography, Box, LinearProgress, Card, CardContent, Grid, Paper } from '@mui/material'

function MetricBox({ label, value }) {
    return (
        <Paper elevation={0} sx={{ p: 1, bgcolor: 'grey.50', display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '100%' }}>
            <Typography variant="body2" color="text.secondary">
                {label}:
            </Typography>
            <Typography variant="body1" fontWeight="medium">
                {value}
            </Typography>
        </Paper>
    )
}

export default function TrainingStatus({ route, downloadModel, setTraining }) {
    const [trainingStatus, setTrainingStatus] = useState(null)
    const socketRef = useRef(null)

    useEffect(() => {
        socketRef.current = io('http://localhost:5000', {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: 10,
            reconnectionDelay: 1000
        })

        function handleSocketData(data) {
            setTrainingStatus(data.status)
            setTraining(true)

            if (data.status['Epoch'] === data.status['Total Epochs']) {
                setTraining(false)

                downloadModel()
            }
        }

        socketRef.current.on(`${route}_training_status`, handleSocketData)

        return () => {
            if (socketRef.current) {
                socketRef.current.off(`${route}_training_status`, handleSocketData)
                socketRef.current.disconnect()
            }
        }
    }, [])

    if (!trainingStatus) {
        return null
    } else {
        return (
            <Card elevation={0} >
                <CardContent>
                    <Box sx={{ mb: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant='body2'>Progress</Typography>
                            <Typography variant='body2'>Epoch {trainingStatus['Epoch']}/{trainingStatus['Total Epochs']}</Typography>
                        </Box>
                        <LinearProgress variant='determinate' value={trainingStatus['Epoch'] / trainingStatus['Total Epochs'] * 100} sx={{ height: 8, borderRadius: 1 }}/>
                    </Box>

                    <Grid container spacing={1}>
                        {Object.entries(trainingStatus.metrics).map(([key, value]) => (
                            <Grid item xs={6} sm={4} md={3} key={key}>
                                <MetricBox label={key} value={value} />
                            </Grid>
                        ))}
                    </Grid>
                </CardContent>
            </Card>

        )
    }
}